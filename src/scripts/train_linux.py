python src/train.py -path2data 'F:/PATSDATASET/speaker/pats/data' -path2outdata 'F:/PATSDATASET/speaker/pats/data' -batch_size 32 -cpk speech2gesture -early_stopping 0 -exp 1 -fs_new '[15, 15]' -gan 1 -input_modalities '["audio/log_mel_400"]' -loss L1Loss -modalities '["pose/normalize", "audio/log_mel_400"]' -model Speech2Gesture_G -note speech2gesture -num_epochs 100 -overfit 0 -render 0 -save_dir save/speech2gesture/speaker -speaker '["oliver", "noah", "seth", "shelly"]' -stop_thresh 3 -tb 1 -window_hop 5