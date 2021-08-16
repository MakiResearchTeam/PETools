from petools.model_tools.one_euro_filter import OneEuroModule
from petools.model_tools.transformers import HumanProcessor, Transformer, PoseTransformer
from petools.model_tools.transformers import Postprocess3D, Postprocess2D, Preprocess3D, Preprocess2D, SequenceBuffer
from petools.model_tools.transformers.utils import H36_2DPOINTS_DIM_FLAT


def init_smoother():
    return lambda: OneEuroModule()


def init_corrector(pb_path: str) -> callable:
    human_processor = HumanProcessor.init_from_lib()
    corrector_t = Transformer(protobuf_path=pb_path)
    corrector_fn = lambda: PoseTransformer(
        transformer=corrector_t,
        seq_buffer=SequenceBuffer(dim=H36_2DPOINTS_DIM_FLAT, seqlen=32),
        preprocess=Preprocess2D(human_processor),
        postprocess=Postprocess2D(human_processor)
    )
    return corrector_fn


def init_converter(pb_path: str) -> callable:
    human_processor = HumanProcessor.init_from_lib()
    converter_t = Transformer(protobuf_path=pb_path)
    converter_fn = lambda: PoseTransformer(
        transformer=converter_t,
        seq_buffer=SequenceBuffer(dim=H36_2DPOINTS_DIM_FLAT, seqlen=32),
        preprocess=Preprocess3D(human_processor),
        postprocess=Postprocess3D(human_processor)
    )
    return converter_fn

