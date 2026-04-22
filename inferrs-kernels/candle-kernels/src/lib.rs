mod ptx {
    include!(concat!(env!("OUT_DIR"), "/ptx.rs"));
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Id {
    Affine,
    Binary,
    Cast,
    Conv,
    Fill,
    FlashAttn,
    Indexing,
    LinearAttn,
    LinearAttnScan,
    Quantized,
    Reduce,
    Sort,
    Ternary,
    Unary,
    IqDequant,
    GptqInt4,
}

pub const ALL_IDS: [Id; 16] = [
    Id::Affine,
    Id::Binary,
    Id::Cast,
    Id::Conv,
    Id::Fill,
    Id::FlashAttn,
    Id::Indexing,
    Id::LinearAttn,
    Id::LinearAttnScan,
    Id::Quantized,
    Id::Reduce,
    Id::Sort,
    Id::Ternary,
    Id::Unary,
    Id::IqDequant,
    Id::GptqInt4,
];

pub struct Module {
    index: usize,
    ptx: &'static str,
}

impl Module {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn ptx(&self) -> &'static str {
        self.ptx
    }
}

const fn module_index(id: Id) -> usize {
    let mut i = 0;
    while i < ALL_IDS.len() {
        if ALL_IDS[i] as u32 == id as u32 {
            return i;
        }
        i += 1;
    }
    panic!("id not found")
}

macro_rules! mdl {
    ($cst:ident, $id:ident) => {
        pub const $cst: Module = Module {
            index: module_index(Id::$id),
            ptx: ptx::$cst,
        };
    };
}

mdl!(AFFINE, Affine);
mdl!(BINARY, Binary);
mdl!(CAST, Cast);
mdl!(CONV, Conv);
mdl!(FILL, Fill);
mdl!(FLASH_ATTN, FlashAttn);
mdl!(INDEXING, Indexing);
mdl!(LINEAR_ATTN, LinearAttn);
mdl!(LINEAR_ATTN_SCAN, LinearAttnScan);
mdl!(QUANTIZED, Quantized);
mdl!(REDUCE, Reduce);
mdl!(SORT, Sort);
mdl!(TERNARY, Ternary);
mdl!(UNARY, Unary);
mdl!(IQ_DEQUANT, IqDequant);
mdl!(GPTQ_INT4, GptqInt4);

pub mod ffi;
