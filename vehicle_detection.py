import helper
import numpy as np
from feature_extraction import *
class FrameQueue:
    """
    This class is used to maintain a queue of heat-map frames
    """

    def __init__(self, max_frames):
        self.frames = []
        self.max_frames = max_frames

    def enqueue(self, frame):
        self.frames.insert(0, frame)

    def _size(self):
        return len(self.frames)

    def _dequeue(self):
        num_element_before = len(self.frames)
        self.frames.pop()
        num_element_after = len(self.frames)

        assert num_element_before == (num_element_after + 1)

    def sum_frames(self):
        if self._size() > self.max_frames:
            self._dequeue()
        all_frames = np.array(self.frames)
        return np.sum(all_frames, axis=0)


class VehicleDetector:
    """
    This is the main class of the project. It encapsulates methods we created for sliding windows, feature generation,
    machine learning mode, and remove duplicates and false positives. Also, internally, it calls other utility methods
    for task such as drawing bounding boxes.
    """

    def __init__(self, color_space, orient, pix_per_cell, cell_per_block,
                 hog_channel, spatial_size, hist_bins, spatial_feat,
                 hist_feat, hog_feat, y_start_stop, x_start_stop, xy_window,
                 xy_overlap, heat_threshold, scaler, classifier,scale,cells_per_step):
        self.color_space = color_space
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.spatial_feat = spatial_feat
        self.hist_feat = hist_feat
        self.hog_feat = hog_feat
        self.y_start_stop = y_start_stop
        self.x_start_stop = x_start_stop
        self.xy_window = xy_window
        self.xy_overlap = xy_overlap
        self.heat_threshold = heat_threshold
        self.scaler = scaler
        self.classifier = classifier
        self.scale = scale
        self.cells_per_step = cells_per_step

        self.frame_queue = FrameQueue(10)

    def detect(self, input_image):
        copy_image = np.copy(input_image)
        copy_image = copy_image.astype(np.float32) / 255.0
        if False:
            slided_windows = slide_window(copy_image, x_start_stop=self.x_start_stop,
                                          y_start_stop=self.y_start_stop,
                                          xy_window=self.xy_window, xy_overlap=self.xy_overlap)

            on_windows = search_windows(copy_image, slided_windows, self.classifier, self.scaler,
                                        color_space=self.color_space, spatial_size=self.spatial_size,
                                        hist_bins=self.hist_bins, orient=self.orient,
                                        pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block,
                                        hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                        hist_feat=self.hist_feat, hog_feat=self.hog_feat)
        else:
            pass
            draw_img, on_windows, windows = find_cars(copy_image, self.classifier, self.scaler, scale = self.scale,
                                             x_start_stop=self.x_start_stop, y_start_stop=self.y_start_stop,
                                             window=self.xy_window[0], cells_per_step = self.cells_per_step,
                                             color_space=self.color_space, spatial_size=self.spatial_size,
                                             hist_bins=self.hist_bins, orient=self.orient,
                                             pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block)
            y_start_stop1 = [self.y_start_stop[0],(self.y_start_stop[1]-100)]
            draw_img, on_windows1, windows = find_cars(copy_image, self.classifier, self.scaler, scale = self.scale-0.5,
                                             x_start_stop=self.x_start_stop, y_start_stop=y_start_stop1,
                                             window=self.xy_window[0], cells_per_step = self.cells_per_step,
                                             color_space=self.color_space, spatial_size=self.spatial_size,
                                             hist_bins=self.hist_bins, orient=self.orient,
                                             pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block)
        heat_map = np.zeros_like(copy_image)
        heat_map = helper.add_heat(heat_map, on_windows)
        heat_map = helper.add_heat(heat_map, on_windows1)
        self.frame_queue.enqueue(heat_map)

        all_frames = self.frame_queue.sum_frames()
        heat_map = helper.apply_threshold(all_frames, self.heat_threshold)

        labels = label(heat_map)

        image_with_bb = helper.draw_labeled_bboxes(input_image, labels)
        return image_with_bb
