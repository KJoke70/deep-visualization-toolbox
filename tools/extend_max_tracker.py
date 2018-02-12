import find_maxes.max_tracker

def output_max_patches_unit(settings, max_tracker, net, layer_name, idx_begin, idx_end, num_top, datadir, filelist, outdir, search_min, do_which):
    do_maxes, do_deconv, do_deconv_norm, do_backprop, do_backprop_norm, do_info = do_which
    assert do_maxes or do_deconv or do_deconv_norm or do_backprop or do_backprop_norm or do_info, 'nothing to do'

    sys.path.insert(0, os.path.join(settings.caffevis_caffe_root, 'python'))
    import caffe

    mt = max_tracker

    locs = mt.min_locs if search_min else mt.max_locs
    vals = mt.min_vals if search_min else mt.max_vals

    image_filenames, image_labels = get_files_list(settings)

    if settings.is_siamese:
        print 'Loaded filenames and labels for %d pairs' % len(image_filenames)
        print '  First pair', image_filenames[0]
    else:
        print 'Loaded filenames and labels for %d files' % len(image_filenames)
        print '  First file', os.path.join(datadir, image_filenames[0])

    siamese_helper = SiameseHelper(settings.layers_list)

    num_top_in_mt = locs.shape[1]
    assert num_top <= num_top_in_mt, 'Requested %d top images but MaxTracker contains only %d' % (num_top, num_top_in_mt)
    assert idx_end >= idx_begin, 'Range error'

    # minor fix for backwards compatability
    if hasattr(mt, 'is_conv'):
        mt.is_spatial = mt.is_conv

    size_ii, size_jj = get_max_data_extent(net, settings, layer_name, mt.is_spatial)
    data_size_ii, data_size_jj = net.blobs['data'].data.shape[2:4]

    net_input_dims = net.blobs['data'].data.shape[2:4]

    # prepare variables used for batches
    batch = [None] * settings.max_tracker_batch_size
    for i in range(0, settings.max_tracker_batch_size):
        batch[i] = MaxTrackerCropBatchRecord()

    batch_index = 0

    channel_to_info_file = dict()

    n_total_images = (idx_end-idx_begin) * num_top
    for cc, channel_idx in enumerate(range(idx_begin, idx_end)):

        unit_dir = os.path.join(outdir, layer_name, 'unit_%04d' % channel_idx)
        mkdir_p(unit_dir)

        # check if all required outputs exist, in which case skip this iteration
        [info_filename,
         maxim_filenames,
         deconv_filenames,
         deconvnorm_filenames,
         backprop_filenames,
         backpropnorm_filenames] = generate_output_names(unit_dir, num_top, do_info, do_maxes, do_deconv, do_deconv_norm, do_backprop, do_backprop_norm, search_min)

        relevant_outputs = info_filename + \
                           maxim_filenames + \
                           deconv_filenames + \
                           deconvnorm_filenames + \
                           backprop_filenames + \
                           backpropnorm_filenames

        # we skip generation if:
        # 1. all outputs exist, AND
        # 2.1.   (not last iteration OR
        # 2.2.    last iteration, but batch is empty)
        relevant_outputs_exist = [os.path.exists(file_name) for file_name in relevant_outputs]
        if all(relevant_outputs_exist) and \
            ((channel_idx != idx_end - 1) or ((channel_idx == idx_end - 1) and (batch_index == 0))):
            print "skipped generation of channel %d in layer %s since files already exist" % (channel_idx, layer_name)
            continue

        if do_info:
            channel_to_info_file[channel_idx] = InfoFileMetadata()
            channel_to_info_file[channel_idx].info_file = open(info_filename[0], 'w')
            channel_to_info_file[channel_idx].ref_count = num_top

            print >> channel_to_info_file[channel_idx].info_file, '# is_spatial val image_idx selected_input_index i(if is_spatial) j(if is_spatial) filename'

        # iterate through maxes from highest (at end) to lowest
        for max_idx_0 in range(num_top):
            batch[batch_index].cc = cc
            batch[batch_index].channel_idx = channel_idx
            batch[batch_index].info_filename = info_filename
            batch[batch_index].maxim_filenames = maxim_filenames
            batch[batch_index].deconv_filenames = deconv_filenames
            batch[batch_index].deconvnorm_filenames = deconvnorm_filenames
            batch[batch_index].backprop_filenames = backprop_filenames
            batch[batch_index].backpropnorm_filenames = backpropnorm_filenames
            batch[batch_index].info_file = channel_to_info_file[channel_idx].info_file

            batch[batch_index].max_idx_0 = max_idx_0
            batch[batch_index].max_idx = num_top_in_mt - 1 - batch[batch_index].max_idx_0

            if mt.is_spatial:

                # fix for backward compatability
                if locs.shape[2] == 5:
                    # remove second column
                    locs = np.delete(locs, 1, 2)

                batch[batch_index].im_idx, batch[batch_index].selected_input_index, batch[batch_index].ii, batch[batch_index].jj = locs[batch[batch_index].channel_idx, batch[batch_index].max_idx]
            else:
                # fix for backward compatability
                if locs.shape[2] == 3:
                    # remove second column
                    locs = np.delete(locs, 1, 2)

                batch[batch_index].im_idx, batch[batch_index].selected_input_index = locs[batch[batch_index].channel_idx, batch[batch_index].max_idx]
                batch[batch_index].ii, batch[batch_index].jj = 0, 0

            # if ii and jj are invalid then there is no data for this "top" image, so we can skip it
            if (batch[batch_index].ii, batch[batch_index].jj) == (-1,-1):
                continue

            batch[batch_index].recorded_val = vals[batch[batch_index].channel_idx, batch[batch_index].max_idx]
            batch[batch_index].filename = image_filenames[batch[batch_index].im_idx]
            do_print = (batch[batch_index].max_idx_0 == 0)
            if do_print:
                print '%s   Output file/image(s) %d/%d   layer %s channel %d' % (datetime.now().ctime(), batch[batch_index].cc * num_top, n_total_images, layer_name, batch[batch_index].channel_idx)


            # print "DEBUG: (mt.is_spatial, batch[batch_index].ii, batch[batch_index].jj, layer_name, size_ii, size_jj, data_size_ii, data_size_jj)", str((mt.is_spatial, batch[batch_index].ii, batch[batch_index].jj, rc, layer_name, size_ii, size_jj, data_size_ii, data_size_jj))

            [batch[batch_index].out_ii_start,
             batch[batch_index].out_ii_end,
             batch[batch_index].out_jj_start,
             batch[batch_index].out_jj_end,
             batch[batch_index].data_ii_start,
             batch[batch_index].data_ii_end,
             batch[batch_index].data_jj_start,
             batch[batch_index].data_jj_end] = \
                compute_data_layer_focus_area(mt.is_spatial, batch[batch_index].ii, batch[batch_index].jj, settings, layer_name,
                                              size_ii, size_jj, data_size_ii, data_size_jj)

            # print "DEBUG: channel:%d out_ii_start:%d out_ii_end:%d out_jj_start:%d out_jj_end:%d data_ii_start:%d data_ii_end:%d data_jj_start:%d data_jj_end:%d" % \
            #       (channel_idx,
            #        batch[batch_index].out_ii_start, batch[batch_index].out_ii_end,
            #        batch[batch_index].out_jj_start, batch[batch_index].out_jj_end,
            #        batch[batch_index].data_ii_start, batch[batch_index].data_ii_end,
            #        batch[batch_index].data_jj_start, batch[batch_index].data_jj_end)

            if do_info:
                print >> batch[batch_index].info_file, 1 if mt.is_spatial else 0, '%.6f' % vals[batch[batch_index].channel_idx, batch[batch_index].max_idx],
                if mt.is_spatial:
                    print >> batch[batch_index].info_file, '%d %d %d %d' % tuple(locs[batch[batch_index].channel_idx, batch[batch_index].max_idx]),
                else:
                    print >> batch[batch_index].info_file, '%d %d' % tuple(locs[batch[batch_index].channel_idx, batch[batch_index].max_idx]),
                print >> batch[batch_index].info_file, batch[batch_index].filename

            if not (do_maxes or do_deconv or do_deconv_norm or do_backprop or do_backprop_norm):
                continue

            with WithTimer('Load image', quiet = not do_print):

                if settings.is_siamese:
                    # in siamese network, filename is a pair of image file names
                    filename1 = batch[batch_index].filename[0]
                    filename2 = batch[batch_index].filename[1]

                    # load both images
                    im1 = caffe.io.load_image(os.path.join(datadir, filename1), color=not settings._calculated_is_gray_model)
                    im2 = caffe.io.load_image(os.path.join(datadir, filename2), color=not settings._calculated_is_gray_model)

                    if settings.siamese_input_mode == 'concat_channelwise':

                        # resize images according to input dimension
                        im1 = resize_without_fit(im1, net_input_dims)
                        im2 = resize_without_fit(im2, net_input_dims)

                        # concatenate channelwise
                        batch[batch_index].im = np.concatenate((im1, im2), axis=2)

                        # convert to float to avoid caffe destroying the image in the scaling phase
                        batch[batch_index].im = batch[batch_index].im.astype(np.float32)

                    elif settings.siamese_input_mode == 'concat_along_width':
                        half_input_dims = (net_input_dims[0], net_input_dims[1] / 2)
                        im1 = resize_without_fit(im1, half_input_dims)
                        im2 = resize_without_fit(im2, half_input_dims)
                        batch[batch_index].im = np.concatenate((im1, im2), axis=1)

                        # convert to float to avoid caffe destroying the image in the scaling phase
                        batch[batch_index].im = batch[batch_index].im.astype(np.float32)

                else:
                    # load image
                    batch[batch_index].im = caffe.io.load_image(os.path.join(datadir, batch[batch_index].filename), color=not settings._calculated_is_gray_model)

                    # resize images according to input dimension
                    batch[batch_index].im = resize_without_fit(batch[batch_index].im, net_input_dims)

                    # convert to float to avoid caffe destroying the image in the scaling phase
                    batch[batch_index].im = batch[batch_index].im.astype(np.float32)

            batch_index += 1

            # if current batch is full
            if batch_index == settings.max_tracker_batch_size \
                    or ((channel_idx == idx_end - 1) and (max_idx_0 == num_top - 1)):  # or last iteration

                with WithTimer('Predict on batch  ', quiet = not do_print):
                    im_batch = [record.im for record in batch]
                    net.predict(im_batch, oversample = False)

                # go over batch and update statistics
                for i in range(0, batch_index):

                    # in siamese network, we wish to return from the normalized layer name and selected input index to the
                    # denormalized layer name, e.g. from "conv1_1" and selected_input_index=1 to "conv1_1_p"
                    batch[i].denormalized_layer_name = siamese_helper.denormalize_layer_name_for_max_tracker(layer_name, batch[i].selected_input_index)
                    batch[i].denormalized_top_name = layer_name_to_top_name(net, batch[i].denormalized_layer_name)
                    batch[i].layer_format = siamese_helper.get_layer_format_by_layer_name(layer_name)

                    if len(net.blobs[batch[i].denormalized_top_name].data.shape) == 4:
                        if settings.is_siamese and batch[i].layer_format == 'siamese_batch_pair':
                            reproduced_val = net.blobs[batch[i].denormalized_top_name].data[batch[i].selected_input_index, batch[i].channel_idx, batch[i].ii, batch[i].jj]

                        else: # normal network, or siamese in siamese_layer_pair format
                            reproduced_val = net.blobs[batch[i].denormalized_top_name].data[i, batch[i].channel_idx, batch[i].ii, batch[i].jj]

                    else:
                        if settings.is_siamese and batch[i].layer_format == 'siamese_batch_pair':
                            reproduced_val = net.blobs[batch[i].denormalized_top_name].data[batch[i].selected_input_index, batch[i].channel_idx]

                        else:  # normal network, or siamese in siamese_layer_pair format
                            reproduced_val = net.blobs[batch[i].denormalized_top_name].data[i, batch[i].channel_idx]

                    if abs(reproduced_val - batch[i].recorded_val) > .1:
                        print 'Warning: recorded value %s is suspiciously different from reproduced value %s. Is the filelist the same?' % (batch[i].recorded_val, reproduced_val)

                    if do_maxes:
                        #grab image from data layer, not from im (to ensure preprocessing / center crop details match between image and deconv/backprop)

                        out_arr = extract_patch_from_image(net.blobs['data'].data[i], net, batch[i].selected_input_index, settings,
                                                           batch[i].data_ii_end, batch[i].data_ii_start, batch[i].data_jj_end, batch[i].data_jj_start,
                                                           batch[i].out_ii_end, batch[i].out_ii_start, batch[i].out_jj_end, batch[i].out_jj_start, size_ii, size_jj)

                        with WithTimer('Save img  ', quiet = not do_print):
                            save_caffe_image(out_arr, batch[i].maxim_filenames[batch[i].max_idx_0],
                                             autoscale = False, autoscale_center = 0)

                if do_deconv or do_deconv_norm:

                    # TODO: we can improve performance by doing batch of deconv_from_layer, but only if we group
                    # together instances which have the same selected_input_index, this can be done by holding two
                    # separate batches

                    for i in range(0, batch_index):
                        diffs = net.blobs[batch[i].denormalized_top_name].diff * 0

                        if settings.is_siamese and batch[i].layer_format == 'siamese_batch_pair':
                            if diffs.shape[0] == 2:
                                if len(diffs.shape) == 4:
                                    diffs[batch[i].selected_input_index, batch[i].channel_idx, batch[i].ii, batch[i].jj] = 1.0
                                else:
                                    # note: the following will not crash, since we already checked we have 2 outputs, so selected_input_index is either 0 or 1
                                    assert batch[i].selected_input_index != -1
                                    diffs[batch[i].selected_input_index, batch[i].channel_idx] = 1.0
                            elif diffs.shape[0] == 1:
                                if len(diffs.shape) == 4:
                                    diffs[0, batch[i].channel_idx, batch[i].ii, batch[i].jj] = 1.0
                                else:
                                    diffs[0, batch[i].channel_idx] = 1.0

                        else: # normal network, or siamese in siamese_layer_pair format
                            if len(diffs.shape) == 4:
                                diffs[i, batch[i].channel_idx, batch[i].ii, batch[i].jj] = 1.0
                            else:
                                diffs[i, batch[i].channel_idx] = 1.0

                        with WithTimer('Deconv    ', quiet = not do_print):
                            net.deconv_from_layer(batch[i].denormalized_layer_name, diffs, zero_higher=True, deconv_type='Guided Backprop')

                        out_arr = extract_patch_from_image(net.blobs['data'].diff[i], net, batch[i].selected_input_index, settings,
                                                           batch[i].data_ii_end, batch[i].data_ii_start, batch[i].data_jj_end, batch[i].data_jj_start,
                                                           batch[i].out_ii_end, batch[i].out_ii_start, batch[i].out_jj_end, batch[i].out_jj_start, size_ii, size_jj)

                        if out_arr.max() == 0:
                            print 'Warning: Deconv out_arr in range', out_arr.min(), 'to', out_arr.max(), 'ensure force_backward: true in prototxt'

                        if do_deconv:
                            with WithTimer('Save img  ', quiet=not do_print):
                                save_caffe_image(out_arr, batch[i].deconv_filenames[batch[i].max_idx_0],
                                                 autoscale=False, autoscale_center=0)
                        if do_deconv_norm:
                            out_arr = np.linalg.norm(out_arr, axis=0)
                            with WithTimer('Save img  ', quiet=not do_print):
                                save_caffe_image(out_arr, batch[i].deconvnorm_filenames[batch[i].max_idx_0])

                if do_backprop or do_backprop_norm:

                    for i in range(0, batch_index):
                        diffs = net.blobs[batch[i].denormalized_top_name].diff * 0

                        if len(diffs.shape) == 4:
                            diffs[i, batch[i].channel_idx, batch[i].ii, batch[i].jj] = 1.0
                        else:
                            diffs[i, batch[i].channel_idx] = 1.0

                    with WithTimer('Backward batch  ', quiet = not do_print):
                        net.backward_from_layer(batch[i].denormalized_layer_name, diffs)

                    for i in range(0, batch_index):

                        out_arr = extract_patch_from_image(net.blobs['data'].diff[i], net, batch[i].selected_input_index, settings,
                                                           batch[i].data_ii_end, batch[i].data_ii_start, batch[i].data_jj_end, batch[i].data_jj_start,
                                                           batch[i].out_ii_end, batch[i].out_ii_start, batch[i].out_jj_end, batch[i].out_jj_start, size_ii, size_jj)

                        if out_arr.max() == 0:
                            print 'Warning: Deconv out_arr in range', out_arr.min(), 'to', out_arr.max(), 'ensure force_backward: true in prototxt'
                        if do_backprop:
                            with WithTimer('Save img  ', quiet = not do_print):
                                save_caffe_image(out_arr, batch[i].backprop_filenames[batch[i].max_idx_0],
                                                 autoscale = False, autoscale_center = 0)
                        if do_backprop_norm:
                            out_arr = np.linalg.norm(out_arr, axis=0)
                            with WithTimer('Save img  ', quiet = not do_print):
                                save_caffe_image(out_arr, batch[i].backpropnorm_filenames[batch[i].max_idx_0])

                # close info files
                for i in range(0, batch_index):
                    channel_to_info_file[batch[i].channel_idx].ref_count -= 1
                    if channel_to_info_file[batch[i].channel_idx].ref_count == 0:
                        if do_info:
                            channel_to_info_file[batch[i].channel_idx].info_file.close()

                batch_index = 0