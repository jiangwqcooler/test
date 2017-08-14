name: "VGG_VOC2007_SSD_97x97_deploy"
input: "data"
input_shape {
  dim: 1
  dim: 3
  dim: 97
  dim: 97
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "Convolution1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    num_output: 32
    bias_term: false
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "msra"
    }
  }
}
layer {
  name: "conv1/bn"
  type: "BatchNorm"
  bottom: "Convolution1"
  top: "BatchNorm1"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
}
layer {
  name: "conv1/scale"
  type: "Scale"
  bottom: "BatchNorm1"
  top: "Scale1"
  scale_param {
    filler {
      value: 1
    }
    bias_term: true
    bias_filler {
      value: 0
    }
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "Scale1"
  top: "conv1"
}
layer {
  name: "conv2_2/dw"
  type: "Convolution"
  bottom: "conv1"
  top: "Convolution2"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    num_output: 32
    bias_term: false
    pad: 0
    kernel_size: 3
    group: 32
    stride: 2
    weight_filler {
      type: "msra"
    }
  }
}
layer {
  name: "conv2_2/dw/bn"
  type: "BatchNorm"
  bottom: "Convolution2"
  top: "BatchNorm2"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
}
layer {
  name: "conv2_2/dw/scale"
  type: "Scale"
  bottom: "BatchNorm2"
  top: "Scale2"
  scale_param {
    filler {
      value: 1
    }
    bias_term: true
    bias_filler {
      value: 0
    }
  }
}
layer {
  name: "relu2_2/dw"
  type: "ReLU"
  bottom: "Scale2"
  top: "conv2_2_dw"
}
layer {
  name: "conv2_2/sep"
  type: "Convolution"
  bottom: "conv2_2_dw"
  top: "Convolution3"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    num_output: 64
    bias_term: false
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
  }
}
layer {
  name: "conv2_2/sep/bn"
  type: "BatchNorm"
  bottom: "Convolution3"
  top: "BatchNorm3"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
}
layer {
  name: "conv2_2/sep/scale"
  type: "Scale"
  bottom: "BatchNorm3"
  top: "Scale3"
  scale_param {
    filler {
      value: 1
    }
    bias_term: true
    bias_filler {
      value: 0
    }
  }
}
layer {
  name: "relu2_2/sep"
  type: "ReLU"
  bottom: "Scale3"
  top: "conv2_2_sep"
}
layer {
  name: "conv3_2/dw"
  type: "Convolution"
  bottom: "conv2_2_sep"
  top: "Convolution4"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    num_output: 64
    bias_term: false
    pad: 0
    kernel_size: 3
    group: 64
    stride: 2
    weight_filler {
      type: "msra"
    }
  }
}
layer {
  name: "conv3_2/dw/bn"
  type: "BatchNorm"
  bottom: "Convolution4"
  top: "BatchNorm4"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
}
layer {
  name: "conv3_2/dw/scale"
  type: "Scale"
  bottom: "BatchNorm4"
  top: "Scale4"
  scale_param {
    filler {
      value: 1
    }
    bias_term: true
    bias_filler {
      value: 0
    }
  }
}
layer {
  name: "relu3_2/dw"
  type: "ReLU"
  bottom: "Scale4"
  top: "conv3_2_dw"
}
layer {
  name: "conv3_2/sep"
  type: "Convolution"
  bottom: "conv3_2_dw"
  top: "Convolution5"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    num_output: 128
    bias_term: false
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
  }
}
layer {
  name: "conv3_2/sep/bn"
  type: "BatchNorm"
  bottom: "Convolution5"
  top: "BatchNorm5"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
}
layer {
  name: "conv3_2/sep/scale"
  type: "Scale"
  bottom: "BatchNorm5"
  top: "Scale5"
  scale_param {
    filler {
      value: 1
    }
    bias_term: true
    bias_filler {
      value: 0
    }
  }
}
layer {
  name: "relu3_2/sep"
  type: "ReLU"
  bottom: "Scale5"
  top: "conv3_2_sep"
}
layer {
  name: "conv4_2/dw"
  type: "Convolution"
  bottom: "conv3_2_sep"
  top: "Convolution6"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    num_output: 128
    bias_term: false
    pad: 0
    kernel_size: 3
    group: 128
    stride: 2
    weight_filler {
      type: "msra"
    }
  }
}
layer {
  name: "conv4_2/dw/bn"
  type: "BatchNorm"
  bottom: "Convolution6"
  top: "BatchNorm6"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
}
layer {
  name: "conv4_2/dw/scale"
  type: "Scale"
  bottom: "BatchNorm6"
  top: "Scale6"
  scale_param {
    filler {
      value: 1
    }
    bias_term: true
    bias_filler {
      value: 0
    }
  }
}
layer {
  name: "relu4_2/dw"
  type: "ReLU"
  bottom: "Scale6"
  top: "conv4_2_dw"
}
layer {
  name: "conv4_2/sep"
  type: "Convolution"
  bottom: "conv4_2_dw"
  top: "Convolution7"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  convolution_param {
    num_output: 256
    bias_term: false
    pad: 0
    kernel_size: 1
    stride: 1
    weight_filler {
      type: "msra"
    }
  }
}
layer {
  name: "conv4_2/sep/bn"
  type: "BatchNorm"
  bottom: "Convolution7"
  top: "BatchNorm7"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
}
layer {
  name: "conv4_2/sep/scale"
  type: "Scale"
  bottom: "BatchNorm7"
  top: "Scale7"
  scale_param {
    filler {
      value: 1
    }
    bias_term: true
    bias_filler {
      value: 0
    }
  }
}
layer {
  name: "relu4_2/sep"
  type: "ReLU"
  bottom: "Scale7"
  top: "conv4_2_sep"
}
layer {
  name: "conv5"
  type: "Convolution"
  bottom: "conv4_2_sep"
  top: "conv5"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 0
    kernel_size: 3
    stride: 2
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv5_relu"
  type: "ReLU"
  bottom: "conv5"
  top: "conv5"
}
layer {
  name: "conv6"
  type: "Convolution"
  bottom: "conv5"
  top: "conv6"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 168
    pad: 0
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv6_relu"
  type: "ReLU"
  bottom: "conv6"
  top: "conv6"
}
layer {
  name: "conv7"
  type: "Convolution"
  bottom: "conv6"
  top: "conv7"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    pad: 0
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv7_relu"
  type: "ReLU"
  bottom: "conv7"
  top: "conv7"
}
layer {
  name: "conv4_2_sep_norm"
  type: "Normalize"
  bottom: "conv4_2_sep"
  top: "conv4_2_sep_norm"
  norm_param {
    across_spatial: false
    scale_filler {
      type: "constant"
      value: 20
    }
    channel_shared: false
  }
}
layer {
  name: "conv4_2_sep_norm_mbox_loc"
  type: "Convolution"
  bottom: "conv4_2_sep_norm"
  top: "conv4_2_sep_norm_mbox_loc"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 16
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv4_2_sep_norm_mbox_loc_perm"
  type: "Permute"
  bottom: "conv4_2_sep_norm_mbox_loc"
  top: "conv4_2_sep_norm_mbox_loc_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv4_2_sep_norm_mbox_loc_flat"
  type: "Flatten"
  bottom: "conv4_2_sep_norm_mbox_loc_perm"
  top: "conv4_2_sep_norm_mbox_loc_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv4_2_sep_norm_mbox_conf"
  type: "Convolution"
  bottom: "conv4_2_sep_norm"
  top: "conv4_2_sep_norm_mbox_conf"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 8
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv4_2_sep_norm_mbox_conf_perm"
  type: "Permute"
  bottom: "conv4_2_sep_norm_mbox_conf"
  top: "conv4_2_sep_norm_mbox_conf_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv4_2_sep_norm_mbox_conf_flat"
  type: "Flatten"
  bottom: "conv4_2_sep_norm_mbox_conf_perm"
  top: "conv4_2_sep_norm_mbox_conf_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv4_2_sep_norm_mbox_priorbox"
  type: "PriorBox"
  bottom: "conv4_2_sep_norm"
  bottom: "data"
  top: "conv4_2_sep_norm_mbox_priorbox"
  prior_box_param {
    min_size: 9.7
    max_size: 19.4
    aspect_ratio: 2
    flip: true
    clip: false
    variance: 0.1
    variance: 0.1
    variance: 0.2
    variance: 0.2
    step: 8
    offset: 0.5
  }
}
layer {
  name: "conv5_mbox_loc"
  type: "Convolution"
  bottom: "conv5"
  top: "conv5_mbox_loc"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 24
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv5_mbox_loc_perm"
  type: "Permute"
  bottom: "conv5_mbox_loc"
  top: "conv5_mbox_loc_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv5_mbox_loc_flat"
  type: "Flatten"
  bottom: "conv5_mbox_loc_perm"
  top: "conv5_mbox_loc_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv5_mbox_conf"
  type: "Convolution"
  bottom: "conv5"
  top: "conv5_mbox_conf"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 12
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv5_mbox_conf_perm"
  type: "Permute"
  bottom: "conv5_mbox_conf"
  top: "conv5_mbox_conf_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv5_mbox_conf_flat"
  type: "Flatten"
  bottom: "conv5_mbox_conf_perm"
  top: "conv5_mbox_conf_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv5_mbox_priorbox"
  type: "PriorBox"
  bottom: "conv5"
  bottom: "data"
  top: "conv5_mbox_priorbox"
  prior_box_param {
    min_size: 19.4
    max_size: 53.35
    aspect_ratio: 2
    aspect_ratio: 3
    flip: true
    clip: false
    variance: 0.1
    variance: 0.1
    variance: 0.2
    variance: 0.2
    step: 16
    offset: 0.5
  }
}
layer {
  name: "conv6_mbox_loc"
  type: "Convolution"
  bottom: "conv6"
  top: "conv6_mbox_loc"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 24
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv6_mbox_loc_perm"
  type: "Permute"
  bottom: "conv6_mbox_loc"
  top: "conv6_mbox_loc_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv6_mbox_loc_flat"
  type: "Flatten"
  bottom: "conv6_mbox_loc_perm"
  top: "conv6_mbox_loc_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv6_mbox_conf"
  type: "Convolution"
  bottom: "conv6"
  top: "conv6_mbox_conf"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 12
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv6_mbox_conf_perm"
  type: "Permute"
  bottom: "conv6_mbox_conf"
  top: "conv6_mbox_conf_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv6_mbox_conf_flat"
  type: "Flatten"
  bottom: "conv6_mbox_conf_perm"
  top: "conv6_mbox_conf_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv6_mbox_priorbox"
  type: "PriorBox"
  bottom: "conv6"
  bottom: "data"
  top: "conv6_mbox_priorbox"
  prior_box_param {
    min_size: 53.35
    max_size: 87.3
    aspect_ratio: 2
    aspect_ratio: 3
    flip: true
    clip: false
    variance: 0.1
    variance: 0.1
    variance: 0.2
    variance: 0.2
    step: 64
    offset: 0.5
  }
}
layer {
  name: "conv7_mbox_loc"
  type: "Convolution"
  bottom: "conv7"
  top: "conv7_mbox_loc"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 16
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv7_mbox_loc_perm"
  type: "Permute"
  bottom: "conv7_mbox_loc"
  top: "conv7_mbox_loc_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv7_mbox_loc_flat"
  type: "Flatten"
  bottom: "conv7_mbox_loc_perm"
  top: "conv7_mbox_loc_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv7_mbox_conf"
  type: "Convolution"
  bottom: "conv7"
  top: "conv7_mbox_conf"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 8
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "conv7_mbox_conf_perm"
  type: "Permute"
  bottom: "conv7_mbox_conf"
  top: "conv7_mbox_conf_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
layer {
  name: "conv7_mbox_conf_flat"
  type: "Flatten"
  bottom: "conv7_mbox_conf_perm"
  top: "conv7_mbox_conf_flat"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "conv7_mbox_priorbox"
  type: "PriorBox"
  bottom: "conv7"
  bottom: "data"
  top: "conv7_mbox_priorbox"
  prior_box_param {
    min_size: 87.3
    max_size: 121.25
    aspect_ratio: 2
    flip: true
    clip: false
    variance: 0.1
    variance: 0.1
    variance: 0.2
    variance: 0.2
    step: 100
    offset: 0.5
  }
}
layer {
  name: "VGG"
  type: "Concat"
  bottom: "conv4_2_sep_norm_mbox_loc_flat"
  bottom: "conv5_mbox_loc_flat"
  bottom: "conv6_mbox_loc_flat"
  bottom: "conv7_mbox_loc_flat"
  top: "VGG"
  concat_param {
    axis: 1
  }
}
layer {
  name: "mbox_conf"
  type: "Concat"
  bottom: "conv4_2_sep_norm_mbox_conf_flat"
  bottom: "conv5_mbox_conf_flat"
  bottom: "conv6_mbox_conf_flat"
  bottom: "conv7_mbox_conf_flat"
  top: "mbox_conf"
  concat_param {
    axis: 1
  }
}
layer {
  name: "mbox_priorbox"
  type: "Concat"
  bottom: "conv4_2_sep_norm_mbox_priorbox"
  bottom: "conv5_mbox_priorbox"
  bottom: "conv6_mbox_priorbox"
  bottom: "conv7_mbox_priorbox"
  top: "mbox_priorbox"
  concat_param {
    axis: 2
  }
}
layer {
  name: "mbox_conf_reshape"
  type: "Reshape"
  bottom: "mbox_conf"
  top: "mbox_conf_reshape"
  reshape_param {
    shape {
      dim: 0
      dim: -1
      dim: 2
    }
  }
}
layer {
  name: "mbox_conf_softmax"
  type: "Softmax"
  bottom: "mbox_conf_reshape"
  top: "mbox_conf_softmax"
  softmax_param {
    axis: 2
  }
}
layer {
  name: "mbox_conf_flatten"
  type: "Flatten"
  bottom: "mbox_conf_softmax"
  top: "mbox_conf_flatten"
  flatten_param {
    axis: 1
  }
}
layer {
  name: "detection_out"
  type: "DetectionOutput"
  bottom: "VGG"
  bottom: "mbox_conf_flatten"
  bottom: "mbox_priorbox"
  top: "detection_out"
  include {
    phase: TEST
  }
  detection_output_param {
    num_classes: 2
    share_location: true
    background_label_id: 0
    nms_param {
      nms_threshold: 0.45
      top_k: 400
    }
    save_output_param {
      output_directory: "/home/wenqiang/code/ssd_118x118_fast_v1/data/VOCdevkit/results/VOC2007/SSD_97x97/Main"
      output_name_prefix: "comp4_det_test_"
      output_format: "VOC"
      label_map_file: "/home/wenqiang/code/caffe/data/VOCdevkit/labelmap_voc.prototxt"
      name_size_file: "/home/wenqiang/code/caffe/data/VOCdevkit/test_name_size.txt"
      num_test_image: 2827
    }
    code_type: CENTER_SIZE
    keep_top_k: 200
    confidence_threshold: 0.01
  }
}

