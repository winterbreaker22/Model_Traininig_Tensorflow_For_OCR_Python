import os
from object_detection.exporter_main_v2 import main
from absl import flags
from absl import app

FLAGS = flags.FLAGS

FLAGS.pipeline_config_path = f'{os.getcwd()}/train_output/pipeline.config'
FLAGS.trained_checkpoint_dir = f'{os.getcwd()}/train_output'
FLAGS.output_directory = f'{os.getcwd()}/exported_model'
FLAGS.input_type = 'image_tensor'
FLAGS.config_override = ''
FLAGS.use_side_inputs = False
FLAGS.side_input_shapes = ''
FLAGS.side_input_types = ''
FLAGS.side_input_names = ''

def export_model(argv):
    flags.mark_flag_as_required('pipeline_config_path')
    flags.mark_flag_as_required('trained_checkpoint_dir')
    flags.mark_flag_as_required('output_directory')

    main(argv)

if __name__ == '__main__':
    app.run(export_model)
