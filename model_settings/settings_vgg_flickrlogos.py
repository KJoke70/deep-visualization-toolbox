
# basic network configuration
base_folder = '%DVT_ROOT%/'
caffevis_deploy_prototxt = base_folder + './models/vgg_flickrlogos/vgg_flickrlogos.prototxt'
caffevis_network_weights = base_folder + './models/vgg_flickrlogos/vgg_flickrlogos.caffemodel'
caffevis_data_mean       = (103.939, 116.779, 123.68)

# input images
static_files_dir = base_folder + './INPUT/'

# UI customization
caffevis_label_layers    = ['fc8_flickrlogos', 'prob']
caffevis_labels          = base_folder + './models/vgg_flickrlogos/flickrlogos_labels.txt'
caffevis_prob_layer      = 'prob'

# Min/max calculation settings
max_tracker_do_histograms = True
max_tracker_do_correlation = False
max_tracker_do_maxes = True
max_tracker_do_deconv = True
max_tracker_do_deconv_norm = Frue
max_tracker_do_backprop = True
max_tracker_do_backprop_norm = True
max_tracker_do_info = True

def caffevis_layer_pretty_name_fn(name):
    return name.replace('conv','c').replace('pool','p')

# offline scripts configuration
# caffevis_outputs_dir = base_folder + './models/vgg_flickrlogos/unit_jpg_vis'
caffevis_outputs_dir = base_folder + './models/vgg_flickrlogos/outputs'
layers_to_output_in_offline_scripts = ['conv5_3']
#layers_to_output_in_offline_scripts = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7', 'fc8_flickrlogos', 'prob']
