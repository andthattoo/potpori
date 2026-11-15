//! The CUDA device: one context, the kernels from `cu/`, and typed device buffers.
//!
//! This is the whole runtime. Kernels are launched by name with raw device pointers and
//! scalars; every launch is synchronous with respect to the next read of its outputs because
//! all work runs on the context's default stream.

use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::marker::PhantomData;
use std::sync::OnceLock;

use cudarc::driver::{result, sys};

/// The kernels, compiled by `build.rs`.
const PTX: &str = concat!(include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx")), "\0");

const KERNELS: [&str; 18] = [
    "embed", "rmsnorm", "gemm_t1", "gemm_t2", "gemm_t4", "gemm_t8", "silu_mul", "rope", "attn_t1",
    "attn_t2", "attn_t4", "attn_t8", "conv2", "gemm4_t1", "gemm4_t2", "gemm4_t4", "gemm4_t8",
    "topk",
];

/// The tensor-core products, compiled for RTX 30-series cards and newer.
const PTX_TC: &str = concat!(
    include_str!(concat!(env!("OUT_DIR"), "/kernels_tc.ptx")),
    "\0"
);

const KERNELS_TC: [&str; 1] = ["gemm4_tc"];

pub struct Gpu {
    ctx: sys::CUcontext,
    funcs: HashMap<&'static str, sys::CUfunction>,
    name: String,
    tensor_cores: bool,
}

// SAFETY: a CUDA context and its function handles may be used from any thread once the
// context is made current there, which every entry point below does.
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

/// The device this process runs on: `POTPORI_GPU` (default 0). Panics if it cannot be opened;
/// nothing in this program works without it.
pub fn gpu() -> &'static Gpu {
    static GPU: OnceLock<Gpu> = OnceLock::new();
    GPU.get_or_init(|| {
        let ordinal = std::env::var("POTPORI_GPU")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        Gpu::open(ordinal).unwrap_or_else(|e| panic!("cannot open CUDA device {ordinal}: {e}"))
    })
}

/// A kernel argument.
#[derive(Clone, Copy)]
pub enum Arg {
    Ptr(u64),
    I32(i32),
    F32(f32),
}

impl From<i32> for Arg {
    fn from(v: i32) -> Self {
        Arg::I32(v)
    }
}
impl From<usize> for Arg {
    fn from(v: usize) -> Self {
        Arg::I32(i32::try_from(v).expect("kernel integer argument out of range"))
    }
}
impl From<f32> for Arg {
    fn from(v: f32) -> Self {
        Arg::F32(v)
    }
}
impl From<u64> for Arg {
    fn from(v: u64) -> Self {
        Arg::Ptr(v)
    }
}

impl Gpu {
    fn open(ordinal: i32) -> Result<Self, result::DriverError> {
        result::init()?;
        let dev = result::device::get(ordinal)?;
        // SAFETY: `dev` is a valid device; the primary context lives for the process.
        let ctx = unsafe { result::primary_ctx::retain(dev)? };
        unsafe { result::ctx::set_current(ctx)? };
        // SAFETY: PTX is NUL-terminated text.
        let module = unsafe { result::module::load_data(PTX.as_ptr() as *const c_void)? };
        let mut funcs = HashMap::new();
        for name in KERNELS {
            let f = unsafe { result::module::get_function(module, CString::new(name).unwrap())? };
            funcs.insert(name, f);
        }
        // SAFETY: `dev` is a valid device.
        let major = unsafe {
            result::device::get_attribute(
                dev,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            )?
        };
        let tensor_cores = major >= 8 && std::env::var_os("POTPORI_NO_TC").is_none();
        if tensor_cores {
            // SAFETY: PTX_TC is NUL-terminated text.
            let module = unsafe { result::module::load_data(PTX_TC.as_ptr() as *const c_void)? };
            for name in KERNELS_TC {
                let f =
                    unsafe { result::module::get_function(module, CString::new(name).unwrap())? };
                funcs.insert(name, f);
            }
        }
        Ok(Self {
            ctx,
            funcs,
            name: format!(
                "{} ({})",
                result::device::get_name(dev)?,
                if tensor_cores {
                    "tensor-core products"
                } else {
                    "scalar products"
                }
            ),
            tensor_cores,
        })
    }

    /// The device's name and which matrix products it runs.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Whether the matrix products run on tensor cores: an RTX 30-series card or newer, and
    /// `POTPORI_NO_TC` unset. They then round their inputs to bf16.
    pub fn tensor_cores(&self) -> bool {
        self.tensor_cores
    }

    /// Free and total device memory, in bytes.
    pub fn memory(&self) -> (usize, usize) {
        self.bind();
        result::mem_get_info().expect("cuMemGetInfo")
    }

    fn bind(&self) {
        // SAFETY: the context is valid for the life of the process.
        unsafe { result::ctx::set_current(self.ctx) }.expect("cuCtxSetCurrent");
    }

    /// Launch `kernel` over `grid` blocks of `block` threads.
    pub fn launch(&self, kernel: &str, grid: (usize, usize), block: usize, args: &[Arg]) {
        if grid.0 == 0 || grid.1 == 0 {
            return;
        }
        self.bind();
        let mut values: Vec<[u8; 8]> = args
            .iter()
            .map(|a| {
                let mut b = [0u8; 8];
                match *a {
                    Arg::Ptr(p) => b.copy_from_slice(&p.to_ne_bytes()),
                    Arg::I32(v) => b[..4].copy_from_slice(&v.to_ne_bytes()),
                    Arg::F32(v) => b[..4].copy_from_slice(&v.to_ne_bytes()),
                }
                b
            })
            .collect();
        let mut params: Vec<*mut c_void> = values
            .iter_mut()
            .map(|v| v.as_mut_ptr() as *mut c_void)
            .collect();
        let f = *self
            .funcs
            .get(kernel)
            .unwrap_or_else(|| panic!("no kernel {kernel}"));
        // SAFETY: the argument list matches the kernel's signature at every call site in
        // `nanbeige::ops`, and the pointers are live device buffers.
        unsafe {
            result::launch_kernel(
                f,
                (grid.0 as u32, grid.1 as u32, 1),
                (block as u32, 1, 1),
                0,
                result::stream::null(),
                &mut params,
            )
        }
        .unwrap_or_else(|e| panic!("launch {kernel}: {e}"));
    }

    /// Wait for everything launched so far. Reads do this themselves; timing code calls it.
    #[cfg(test)]
    pub fn sync(&self) {
        self.bind();
        result::ctx::synchronize().expect("cuCtxSynchronize");
    }
}

/// A device buffer of `len` elements of `T`.
pub struct Buf<T> {
    ptr: sys::CUdeviceptr,
    len: usize,
    _t: PhantomData<T>,
}

// SAFETY: a device pointer is usable from any thread with the context current.
unsafe impl<T> Send for Buf<T> {}

impl<T: Copy> Buf<T> {
    /// Uninitialised-looking but zeroed storage.
    pub fn zeros(len: usize) -> Self {
        gpu().bind();
        let bytes = (len * size_of::<T>()).max(1);
        // SAFETY: plain allocation; zeroed before use.
        let ptr = unsafe { result::malloc_sync(bytes) }.unwrap_or_else(|e| {
            let (free, total) = result::mem_get_info().unwrap_or((0, 0));
            panic!(
                "out of GPU memory allocating {:.2} GiB ({:.2} of {:.2} GiB free): {e}",
                bytes as f64 / (1u64 << 30) as f64,
                free as f64 / (1u64 << 30) as f64,
                total as f64 / (1u64 << 30) as f64
            )
        });
        unsafe { result::memset_d8_sync(ptr, 0, bytes) }.expect("cuMemsetD8");
        Self {
            ptr,
            len,
            _t: PhantomData,
        }
    }

    pub fn from_slice(data: &[T]) -> Self {
        let buf = Self::zeros(data.len());
        buf.write(0, data);
        buf
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// Device address of element `offset`.
    pub fn at(&self, offset: usize) -> u64 {
        assert!(offset <= self.len, "device offset out of range");
        self.ptr + (offset * size_of::<T>()) as u64
    }

    pub fn ptr(&self) -> u64 {
        self.ptr
    }

    pub fn write(&self, offset: usize, data: &[T]) {
        assert!(offset + data.len() <= self.len, "device write out of range");
        if data.is_empty() {
            return;
        }
        gpu().bind();
        // SAFETY: the range was checked; `T: Copy` is plain data.
        unsafe { result::memcpy_htod_sync(self.at(offset), data) }.expect("cuMemcpyHtoD");
    }

    /// Copy `len` elements from `offset` to the host. Waits for pending kernels.
    pub fn read(&self, offset: usize, len: usize) -> Vec<T>
    where
        T: Default,
    {
        assert!(offset + len <= self.len, "device read out of range");
        let mut out = vec![T::default(); len];
        if len > 0 {
            gpu().bind();
            // SAFETY: the range was checked.
            unsafe { result::memcpy_dtoh_sync(&mut out, self.at(offset)) }.expect("cuMemcpyDtoH");
        }
        out
    }
}

impl<T> Drop for Buf<T> {
    fn drop(&mut self) {
        gpu().bind();
        // SAFETY: allocated by `malloc_sync`, freed once. Pending kernels that use the buffer
        // are finished first.
        unsafe {
            let _ = result::ctx::synchronize();
            let _ = result::free_sync(self.ptr);
        }
    }
}
