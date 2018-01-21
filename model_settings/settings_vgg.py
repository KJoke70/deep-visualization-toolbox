
# basic network configuration
base_folder = '%DVT_ROOT%/'
caffevis_deploy_prototxt = base_folder + './models/vgg/vgg16.prototxt'
caffevis_network_weights = base_folder + './models/vgg/vgg16.caffemodel'
caffevis_data_mean       = (103.939, 116.779, 123.68)

# input images
static_files_dir = base_folder + './INPUT/'

# UI customization
caffevis_label_layers    = ['fc8', 'prob']
caffevis_labels          = base_folder + './models/vgg/imagenet_labels.txt'
caffevis_prob_layer      = 'prob'

def caffevis_layer_pretty_name_fn(name):
    return name.replace('conv','c').replace('pool','p')

# offline scripts configuration
# caffevis_outputs_dir = base_folder + './models/vgg/unit_jpg_vis'
caffevis_outputs_dir = base_folder + './models/vgg/outputs'
layers_to_output_in_offline_scripts = ['conv5_3']
#layers_to_output_in_offline_scripts = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7', 'fc8', 'prob']
