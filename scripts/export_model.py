from object_detection.exporter_main_v2 import main
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('pipeline_config_path', f'{os.getcwd()}/model/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config', 
                    'Path to pipeline config file.')
flags.DEFINE_string('trained_checkpoint_dir', f'{os.getcwd()}/train_output', 
                    'Path to trained checkpoint directory.')
flags.DEFINE_string('output_directory', f'{os.getcwd()}/exported_model', 
                    'Path to output directory for exported model.')
flags.DEFINE_string('input_type', 'image_tensor', 
                    'Input type (image_tensor, tf_example, etc.).')
flags.DEFINE_string('config_override', '', 
                    'Pipeline config overrides as a protobuf text string.')
flags.DEFINE_boolean('use_side_inputs', False, 
                     'Whether to use side inputs.')
flags.DEFINE_string('side_input_shapes', '', 
                    'Side input shapes if using side inputs.')
flags.DEFINE_string('side_input_types', '', 
                    'Side input types if using side inputs.')
flags.DEFINE_string('side_input_names', '', 
                    'Side input names if using side inputs.')

flags.mark_flag_as_required('pipeline_config_path')
flags.mark_flag_as_required('trained_checkpoint_dir')
flags.mark_flag_as_required('output_directory')

def export_model():
    main(None)

if __name__ == '__main__':
    export_model()
