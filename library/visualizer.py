"""
Designed for data visualization, abbr. VIS
PyQtGraph-based replacement for matplotlib - processes every frame without dropping.

Drop-in replacement: same __init__ signature, same run() entry point,
same queue consumption, same filtering pipeline.
"""

import queue
import time
import sys
from datetime import datetime
from multiprocessing import Manager

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtWidgets

from library import SNRV_filter, np_filter
from library.frame_post_processor import FramePProcessor

# Color presets (RGBA float 0-1) matching original colormap intent
RP_COLORS = [
    (0.8, 0.2, 0.8, 1.0),   # C5 - purple-ish
    (0.5, 0.5, 0.5, 1.0),   # C7 - grey
    (0.6, 0.8, 0.2, 1.0),   # C8 - olive
]
RADAR_POS_COLOR = (0.55, 0.0, 0.0, 1.0)       # DarkRed
ENVELOPE_COLOR = (0.98, 0.5, 0.45, 1.0)        # salmon
BGN_COLOR = (0.0, 0.5, 0.0, 1.0)              # green
# Object status colors: grey, green, gold, red
OS_COLORS = [
    (0.5, 0.5, 0.5, 1.0),
    (0.0, 0.5, 0.0, 1.0),
    (1.0, 0.84, 0.0, 1.0),
    (1.0, 0.0, 0.0, 1.0),
]


class Visualizer:
    def __init__(self, run_flag, radar_rd_queue_list, shared_param_dict, **kwargs_CFG):
        """
        get shared values and queues
        """
        self.run_flag = run_flag
        # radar rawdata queue list
        self.radar_rd_queue_list = radar_rd_queue_list
        # shared params
        try:
            self.save_queue = shared_param_dict['save_queue']
        except:
            self.save_queue = Manager().Queue(maxsize=0)
        self.mansave_flag = shared_param_dict['mansave_flag']
        self.autosave_flag = shared_param_dict['autosave_flag']
        self.status = shared_param_dict['proc_status_dict']
        self.status['Module_VIS'] = True

        """
        pass config static parameters
        """
        """ module own config """
        VIS_CFG = kwargs_CFG['VISUALIZER_CFG']
        self.dimension = VIS_CFG['dimension']
        self.VIS_xlim = VIS_CFG['VIS_xlim']
        self.VIS_ylim = VIS_CFG['VIS_ylim']
        self.VIS_zlim = VIS_CFG['VIS_zlim']

        self.auto_inactive_skip_frame = VIS_CFG['auto_inactive_skip_frame']

        """ other configs """
        self.MANSAVE_ENABLE = kwargs_CFG['MANSAVE_ENABLE']
        self.AUTOSAVE_ENABLE = kwargs_CFG['AUTOSAVE_ENABLE']
        self.RDR_CFG_LIST = kwargs_CFG['RADAR_CFG_LIST']

        """
        self content
        """
        self.fpp = FramePProcessor(**kwargs_CFG)  # call other class

        # ---- PyQtGraph setup ----
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication(sys.argv)

        if self.dimension in ('2D', '3D'):
            self._setup_3d_view()  # use 3D view for both (2D just locks camera)
        
        self._log('Start...')

    #  PyQtGraph 3D scene setup
    def _setup_3d_view(self):
        """Create the OpenGL 3D scatter view with grid and axes."""
        self.widget = gl.GLViewWidget()
        self.widget.setWindowTitle('Radar Visualizer (PyQtGraph)')
        self.widget.setGeometry(30, 30, 900, 700)

        # Camera defaults
        if self.dimension == '2D':
            # Top-down view for 2D mode
            self.widget.setCameraPosition(distance=8, elevation=90, azimuth=-90)
        else:
            self.widget.setCameraPosition(distance=8, elevation=25, azimuth=45)

        # Add grid on the XY plane (ground)
        grid = gl.GLGridItem()
        grid.setSize(
            x=self.VIS_xlim[1] - self.VIS_xlim[0],
            y=self.VIS_ylim[1] - self.VIS_ylim[0],
        )
        grid.setSpacing(x=1, y=1)
        grid.translate(
            (self.VIS_xlim[0] + self.VIS_xlim[1]) / 2,
            (self.VIS_ylim[0] + self.VIS_ylim[1]) / 2,
            self.VIS_zlim[0],
        )
        self.widget.addItem(grid)

        # Add axis lines for reference
        axis = gl.GLAxisItem()
        axis.setSize(x=2, y=2, z=2)
        self.widget.addItem(axis)

        # ---- Persistent scatter plot items (updated every frame, never recreated) ----

        # Radar position markers (static, one per radar)
        self.radar_pos_items = []
        for RDR_CFG in self.RDR_CFG_LIST:
            pos = np.array([[
                RDR_CFG['pos_offset'][0],
                RDR_CFG['pos_offset'][1],
                RDR_CFG['pos_offset'][2],
            ]], dtype=np.float32)
            item = gl.GLScatterPlotItem(
                pos=pos, color=RADAR_POS_COLOR, size=12, pxMode=True
            )
            self.widget.addItem(item)
            self.radar_pos_items.append(item)

        # Filtered point cloud scatter (main display)
        self.cloud_scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=(0.8, 0.2, 0.8, 0.8),
            size=4,
            pxMode=True,
        )
        self.widget.addItem(self.cloud_scatter)

        # DBSCAN envelope lines (reuse a pool of line items)
        self._envelope_pool = []
        self._envelope_active = 0

        # BGN area lines
        self._bgn_pool = []
        self._bgn_active = 0

        # Tracking object markers
        self._track_pool = []
        self._track_active = 0

        self.widget.show()

    #  Reusable GL item pools (avoids add/remove overhead every frame)
    def _get_line_item(self, pool, index):
        """Return a GLLinePlotItem from pool, creating if needed."""
        while len(pool) <= index:
            item = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(1, 1, 1, 0), width=1)
            item.setVisible(False)
            self.widget.addItem(item)
            pool.append(item)
        return pool[index]

    def _get_track_item(self, index):
        """Return a GLScatterPlotItem for tracking markers."""
        while len(self._track_pool) <= index:
            item = gl.GLScatterPlotItem(
                pos=np.zeros((1, 3)), color=(0.5, 0.5, 0.5, 1), size=14, pxMode=True
            )
            item.setVisible(False)
            self.widget.addItem(item)
            self._track_pool.append(item)
        return self._track_pool[index]

    def _get_box_item(self, index):
        """Return a GLLinePlotItem for bounding box, creating if needed."""
        while len(self._box_pool) <= index:
            item = gl.GLLinePlotItem(
                pos=np.zeros((2, 3), dtype=np.float32),
                color=(1, 1, 1, 1),
                width=2,
                mode='line_strip'  # important for connected edges
            )
            item.setVisible(False)
            self.widget.addItem(item)
            self._box_pool.append(item)
        return self._box_pool[index]

    def _hide_pool_from(self, pool, start_index):
        """Hide all pool items from start_index onward."""
        for i in range(start_index, len(pool)):
            pool[i].setVisible(False)

    #  Main loop
    def run(self):
        if self.dimension in ('2D', '3D'):
            # Use a QTimer to drive frame updates inside the Qt event loop
            self._timer = QtCore.QTimer()
            self._timer.timeout.connect(self._update_frame)
            self._timer.start(1)  # 1 ms interval = as fast as possible

            # Also check run_flag periodically
            self._flag_timer = QtCore.QTimer()
            self._flag_timer.timeout.connect(self._check_run_flag)
            self._flag_timer.start(100)

            # Run the Qt event loop (blocks here)
            self.app.exec()
        else:
            # No-display mode: just drain queues
            while self.run_flag.value:
                for q in self.radar_rd_queue_list:
                    try:
                        _ = q.get(block=True, timeout=5)
                    except queue.Empty:
                        pass

    def _check_run_flag(self):
        """Periodic check — stop Qt loop when system is shutting down."""
        if not self.run_flag.value:
            self._timer.stop()
            self._flag_timer.stop()
            self.app.quit()

    #  Per-frame update
    def _update_frame(self):
        """Called by QTimer — consumes one frame from each radar queue and renders."""
        if not self.run_flag.value:
            return
        val_data_allradar = np.ndarray([0, 5], dtype=np.float16)
        SNR_noise_allradar = np.ndarray([0, 5], dtype=np.float16)
        save_data_frame = {}

        try:
            # adaptive short skip when no object is detected
            if self.AUTOSAVE_ENABLE and not self.autosave_flag.value:
                for _ in range(self.auto_inactive_skip_frame):
                    for q in self.radar_rd_queue_list:
                        _ = q.get(block=True, timeout=5)

            for i, RDR_CFG in enumerate(self.RDR_CFG_LIST):
                data_1radar = self.radar_rd_queue_list[i].get(block=True, timeout=0.1)

                # apply SNR and speed filter for each radar channel
                val_data, SNR_noise = SNRV_filter(data_1radar, RDR_CFG['SNRV_threshold'])
                val_data_allradar = np.concatenate([val_data_allradar, val_data])
                SNR_noise_allradar = np.concatenate([SNR_noise_allradar, SNR_noise])
                # save the frames
                save_data_frame[RDR_CFG['name']] = data_1radar

                ##NEWLY ADDED
                # Average SNR of valid (SNR-filtered) points for this radar
                if len(val_data) > 0:
                    avg_snr = float(np.mean(val_data[:, 4]))
                    print(f"  {RDR_CFG['name']} valid avg SNR: {avg_snr:.1f}  (n={len(val_data)})")
                else:
                    print(f"  {RDR_CFG['name']} valid avg SNR: N/A  (no points)")

            # Confirm all radars sent data this frame
            names = list(save_data_frame.keys())
            points = [save_data_frame[k].shape[0] for k in names]
            print(f"[VIS] All {len(names)} radars OK: {names} | points per radar: {points}")

        except queue.Empty:
            self._log('Raw Data Queue Empty.')
            return

        # put frame and time into save queue
        self.save_queue.put({
            'source': 'radar',
            'data': save_data_frame,
            'timestamp': time.time(),
        })

        # apply global boundary filter
        val_data_allradar = self.fpp.FPP_boundary_filter(val_data_allradar)
        SNR_noise_allradar = self.fpp.FPP_boundary_filter(SNR_noise_allradar)
        # apply global energy strength filter
        val_data_allradar, global_SNR_noise = self.fpp.FPP_SNRV_filter(val_data_allradar)
        # apply background noise filter
        val_data_allradar = self.fpp.BGN_filter(val_data_allradar)


        # draw signal energy strength
        # for i in range(len(SNR_colormap)):
        #     val_data_allradar_SNR, _ = np_filter(val_data_allradar, idx=4, range_lim=(i * 100, (i + 1) * 100))
        #     self._plot(ax1, val_data_allradar_SNR[:, 0], val_data_allradar_SNR[:, 1], val_data_allradar_SNR[:, 2], marker='.', color=SNR_colormap[i])

        # ---- Update point cloud scatter ----
        if len(val_data_allradar) > 0:
            self.cloud_scatter.setData(
                pos=val_data_allradar[:, :3].astype(np.float32),
                color=(0.8, 0.2, 0.8, 0.8),
                size=4,
            )
        else:
            self.cloud_scatter.setData(pos=np.zeros((1, 3), dtype=np.float32),
                                       color=(0, 0, 0, 0), size=0)

        # run DBSCAN clustering (boxes drawn from tracker below, not raw DBSCAN)
        # vertices_list, valid_points_list, _, DBS_noise = self.fpp.DBS(val_data_allradar)
        vertices_list, valid_points_list, _, DBS_noise = self.fpp.DBS_dynamic_SNR(val_data_allradar)
        env_idx = 0
        for vertices in vertices_list:
            if len(vertices) >= 2:
                item = self._get_line_item(self._envelope_pool, env_idx)
                item.setData(
                    pos=vertices[:, :3].astype(np.float32),
                    color=ENVELOPE_COLOR,
                    width=2,
                )
                item.setVisible(False)
                env_idx += 1
        self._hide_pool_from(self._envelope_pool, env_idx)

        # ---- Background noise filter ----
        if self.fpp.BGN_enable:
            if len(vertices_list) > 0:
                self.fpp.BGN_update(np.concatenate([SNR_noise_allradar, global_SNR_noise, DBS_noise]))
            else:
                self.fpp.BGN_update(np.concatenate([SNR_noise_allradar, global_SNR_noise]))
            # draw BGN area
            BGN_block_list = self.fpp.BGN_get_filter_area()
            bgn_idx = 0
            for bgn in BGN_block_list:
                if len(bgn) >= 2:
                    item = self._get_line_item(self._bgn_pool, bgn_idx)
                    item.setData(
                        pos=bgn[:, :3].astype(np.float32),
                        color=BGN_COLOR,
                        width=1.5,
                    )
                    item.setVisible(True)
                    bgn_idx += 1
            self._hide_pool_from(self._bgn_pool, bgn_idx)

        # tracking system
        if self.fpp.TRK_enable:
            self.fpp.TRK_update_poss_matrix(valid_points_list)
            obj_status_list = []
            trk_idx = 0
            box_idx = 0
            for person in self.fpp.TRK_people_list:
                obj_cp, obj_size, obj_status = person.get_info()
                obj_status_list.append(obj_status)
                # --- Bounding box ---
                if obj_status >= 1:
                    half = obj_size[0] / 2
                    box_vertices = cubehull(
                        None,
                        (obj_cp[0, 0] - half[0], obj_cp[0, 0] + half[0]),
                        (obj_cp[0, 1] - half[1], obj_cp[0, 1] + half[1]),
                        (obj_cp[0, 2] - half[2], obj_cp[0, 2] + half[2]),
                    )
                    box_item = self._get_box_item(box_idx)
                    box_item.setData(
                        pos=box_vertices[:, :3].astype(np.float32),
                        color=OS_COLORS[obj_status],
                        width=2,
                    )
                    box_item.setVisible(True)
                    box_idx += 1

                item = self._get_track_item(trk_idx)
                item.setData(
                    pos=obj_cp[:, :3].astype(np.float32),
                    color=OS_COLORS[obj_status],
                    size=14,
                )
                item.setVisible(True)
                trk_idx += 1
            self._hide_pool_from(self._track_pool, trk_idx)
            self._hide_pool_from(self._box_pool, box_idx)
            # auto save based on object detection
            if self.AUTOSAVE_ENABLE:
                if len(obj_status_list) > 0 and max(obj_status_list) >= 0:
                    self.autosave_flag.value = True
                else:
                    self.autosave_flag.value = False

    def keyPressEvent(self, event):
        """Override if widget is subclassed; here we install an event filter."""
        key = event.key()
        if key == QtCore.Qt.Key_Escape:
            self.run_flag.value = False
        if self.MANSAVE_ENABLE:
            if key == QtCore.Qt.Key_Plus:
                self.mansave_flag.value = 'image'
            elif key == QtCore.Qt.Key_0:
                self.mansave_flag.value = 'video'

    def _log(self, txt):  # print with device name
        print(f'[{self.__class__.__name__}]\t{txt}')

    def __del__(self):
        try:
            self.widget.close()
        except:
            pass
        self._log(f"Closed. Timestamp: {datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
        self.status['Module_VIS'] = False
        self.run_flag.value = False