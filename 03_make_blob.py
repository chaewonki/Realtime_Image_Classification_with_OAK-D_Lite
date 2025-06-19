import shutil
import blobconverter

blob_path = blobconverter.from_openvino(
    xml="./outputs/saved_model.xml",
    bin="./outputs/saved_model.bin",
    data_type="FP16",
    shaves=4,
    version="2022.1",
    use_cache=False
)

shutil.copy(blob_path, "./outputs/hand_model.blob")