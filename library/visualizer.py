"""
Modified Visualizer with TI Visualizer Integration - V2
========================================================
This version allows you to choose WHAT data to send to TI Visualizer:
- 'filtered_pc'  : Point cloud after filtering (before DBSCAN)
- 'clustered_pc' : Only valid points from DBSCAN clusters
- 'tracked_obj'  : Only tracked object centroids
- 'all'          : Send all stages as separate TLVs (advanced)

Author: DarkSZChao (Original), Extended for TI Visualizer Integration V2
"""

import math
import queue
import time
from datetime import datetime
from multiprocessing import Manager

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator

from library import SNRV_filter, np_filter
from library.frame_post_processor import FramePProcessor

# Import TI Visualizer Bridge
try:
    from library.ti_visualizer_bridge import TIVisualizerBridge
    TI_BRIDGE_AVAILABLE = True
except ImportError:
    try:
        from library.ti_visualizer_bridge import TIVisualizerBridge
        TI_BRIDGE_AVAILABLE = True
    except ImportError:
        TI_BRIDGE_AVAILABLE = False
        print("[Warning] TI Visualizer Bridge not available.")

RP_colormap = ['C5', 'C7', 'C8']
SNR_colormap = ['lavender', 'thistle', 'violet', 'darkorchid', 'indigo']
OS_colormap = ['grey', 'green', 'gold', 'red']


class VisualizerTI:
    """
    Enhanced Visualizer with TI Visualizer support.
    
    New configuration options in VISUALIZER_CFG:
        - 'use_ti_visualizer': True/False
        - 'use_matplotlib': True/False  
        - 'ti_port_write': '/tmp/ttyVUSB0'
        - 'ti_port_read': '/tmp/ttyVUSB1'
        - 'ti_output_stage': 'filtered_pc' | 'clustered_pc' | 'tracked_obj' | 'all'
    """
    
    def __init__(self, run_flag, radar_rd_queue_list, shared_param_dict, **kwargs_CFG):
        """
        get shared values and queues
        """
        self.run_flag = run_flag
        self.radar_rd_queue_list = radar_rd_queue_list
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
        VIS_CFG = kwargs_CFG['VISUALIZER_CFG']
        self.dimension = VIS_CFG['dimension']
        self.VIS_xlim = VIS_CFG['VIS_xlim']
        self.VIS_ylim = VIS_CFG['VIS_ylim']
        self.VIS_zlim = VIS_CFG['VIS_zlim']
        self.auto_inactive_skip_frame = VIS_CFG['auto_inactive_skip_frame']

        # TI Visualizer settings
        self.use_ti_visualizer = VIS_CFG.get('use_ti_visualizer', True)
        self.use_matplotlib = VIS_CFG.get('use_matplotlib', False)
        self.ti_port_write = VIS_CFG.get('ti_port_write', '/tmp/ttyUSB0')
        self.ti_port_read = VIS_CFG.get('ti_port_read', '/tmp/ttyUSB1')
        # Options: 'filtered_pc', 'clustered_pc', 'tracked_obj', 'all'
        self.ti_output_stage = VIS_CFG.get('ti_output_stage', 'clustered_pc')

        self.MANSAVE_ENABLE = kwargs_CFG['MANSAVE_ENABLE']
        self.AUTOSAVE_ENABLE = kwargs_CFG['AUTOSAVE_ENABLE']
        self.RDR_CFG_LIST = kwargs_CFG['RADAR_CFG_LIST']

        """
        self content
        """
        self.fpp = FramePProcessor(**kwargs_CFG)

        # Initialize TI Visualizer Bridge
        self.ti_bridge = None
        if self.use_ti_visualizer and TI_BRIDGE_AVAILABLE:
            self.ti_bridge = TIVisualizerBridge(
                port_write=self.ti_port_write,
                port_read=self.ti_port_read,
                fps=20
            )
            try:
                self.ti_bridge.start()
                self._log(f"TI Visualizer Bridge started")
                self._log(f"  Output stage: {self.ti_output_stage}")
                self._log(f"  Connect TI Visualizer to: {self.ti_port_read}")
            except Exception as e:
                self._log(f"Failed to start TI Bridge: {e}")
                self.ti_bridge = None

        # Setup matplotlib (if enabled)
        self.fig = None
        if self.use_matplotlib:
            matplotlib.use('TkAgg')
            plt.rcParams['toolbar'] = 'None'
            self.fig = plt.figure()
            mngr = plt.get_current_fig_manager()
            mngr.window.wm_geometry('+30+30')
            win = plt.gcf().canvas.manager.window
            win.overrideredirect(1)
            plt.ion()

        self._log('Start...')

    def run(self):
        if self.use_matplotlib and self.dimension == '2D':
            ax1 = self.fig.add_subplot(111)
            while self.run_flag.value:
                plt.cla()
                ax1.set_xlim(self.VIS_xlim[0], self.VIS_xlim[1])
                ax1.set_ylim(self.VIS_ylim[0], self.VIS_ylim[1])
                ax1.xaxis.set_major_locator(LinearLocator(5))
                ax1.yaxis.set_major_locator(LinearLocator(5))
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')
                ax1.set_title('Radar')
                self._update_canvas(ax1)

        elif self.use_matplotlib and self.dimension == '3D':
            ax1 = self.fig.add_subplot(111, projection='3d')
            spin = 0
            while self.run_flag.value:
                plt.cla()
                ax1.set_xlim(self.VIS_xlim[0], self.VIS_xlim[1])
                ax1.set_ylim(self.VIS_ylim[0], self.VIS_ylim[1])
                ax1.set_zlim(self.VIS_zlim[0], self.VIS_zlim[1])
                ax1.xaxis.set_major_locator(LinearLocator(3))
                ax1.yaxis.set_major_locator(LinearLocator(3))
                ax1.zaxis.set_major_locator(LinearLocator(3))
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')
                ax1.set_zlabel('z')
                ax1.set_title('Radar')
                spin += 0.04
                ax1.view_init(ax1.elev - 0.5 * math.sin(spin), ax1.azim - 0.3 * math.sin(1.5 * spin))
                self._update_canvas(ax1)
        else:
            # TI Visualizer only mode
            while self.run_flag.value:
                self._update_canvas(None)

    def _update_canvas(self, ax1):
        # Draw radar positions (matplotlib only)
        if ax1 is not None:
            for RDR_CFG in self.RDR_CFG_LIST:
                self._plot(ax1, [RDR_CFG['pos_offset'][0]], [RDR_CFG['pos_offset'][1]], 
                          [RDR_CFG['pos_offset'][2]], marker='o', color='DarkRed')

        # ================================================================
        # STAGE 1: Get raw data from queues
        # ================================================================
        val_data_allradar = np.ndarray([0, 5], dtype=np.float16)
        SNR_noise_allradar = np.ndarray([0, 5], dtype=np.float16)
        save_data_frame = {}
        
        try:
            if self.AUTOSAVE_ENABLE and not self.autosave_flag.value:
                for _ in range(self.auto_inactive_skip_frame):
                    for q in self.radar_rd_queue_list:
                        _ = q.get(block=True, timeout=5)

            for i, RDR_CFG in enumerate(self.RDR_CFG_LIST):
                data_1radar = self.radar_rd_queue_list[i].get(block=True, timeout=5)
                val_data, SNR_noise = SNRV_filter(data_1radar, RDR_CFG['SNRV_threshold'])
                val_data_allradar = np.concatenate([val_data_allradar, val_data])
                SNR_noise_allradar = np.concatenate([SNR_noise_allradar, SNR_noise])
                save_data_frame[RDR_CFG['name']] = data_1radar

        except queue.Empty:
            self._log('Raw Data Queue Empty.')
            self.run_flag.value = False
            return

        # Save to queue
        self.save_queue.put({
            'source': 'radar',
            'data': save_data_frame,
            'timestamp': time.time(),
        })

        # ================================================================
        # STAGE 2: Apply filters
        # ================================================================
        val_data_allradar = self.fpp.FPP_boundary_filter(val_data_allradar)
        SNR_noise_allradar = self.fpp.FPP_boundary_filter(SNR_noise_allradar)
        val_data_allradar, global_SNR_noise = self.fpp.FPP_SNRV_filter(val_data_allradar)
        val_data_allradar = self.fpp.BGN_filter(val_data_allradar)

        # >>> TI OUTPUT OPTION: filtered_pc <<<
        if self.ti_bridge and self.ti_output_stage == 'filtered_pc':
            self.ti_bridge.send_frame(val_data_allradar)

        # ================================================================
        # STAGE 3: DBSCAN Clustering
        # ================================================================
        # Draw SNR colormap (matplotlib)
        if ax1 is not None:
            for i in range(len(SNR_colormap)):
                val_data_allradar_SNR, _ = np_filter(val_data_allradar, idx=4, range_lim=(i * 100, (i + 1) * 100))
                self._plot(ax1, val_data_allradar_SNR[:, 0], val_data_allradar_SNR[:, 1], 
                          val_data_allradar_SNR[:, 2], marker='.', color=SNR_colormap[i])

        # DBSCAN clustering
        vertices_list, valid_points_list, valid_points_total, DBS_noise = self.fpp.DBS_dynamic_SNR(val_data_allradar)

        # >>> TI OUTPUT OPTION: clustered_pc <<<
        if self.ti_bridge and self.ti_output_stage == 'clustered_pc':
            # Send only the valid clustered points (human-detected points)
            if len(valid_points_total) > 0:
                self.ti_bridge.send_frame(valid_points_total)
            else:
                self.ti_bridge.send_frame(np.zeros((0, 5)))

        # Draw DBSCAN envelopes (matplotlib)
        if ax1 is not None:
            for vertices in vertices_list:
                self._plot(ax1, vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                          linestyle='-', color='salmon')

        # ================================================================
        # STAGE 4: Background Noise Update
        # ================================================================
        if self.fpp.BGN_enable:
            if len(vertices_list) > 0:
                self.fpp.BGN_update(np.concatenate([SNR_noise_allradar, global_SNR_noise, DBS_noise]))
            else:
                self.fpp.BGN_update(np.concatenate([SNR_noise_allradar, global_SNR_noise]))
            
            if ax1 is not None:
                BGN_block_list = self.fpp.BGN_get_filter_area()
                for bgn in BGN_block_list:
                    self._plot(ax1, bgn[:, 0], bgn[:, 1], bgn[:, 2], 
                              marker='.', linestyle='-', color='g')

        # ================================================================
        # STAGE 5: Human Tracking
        # ================================================================
        tracked_points = np.ndarray([0, 5], dtype=np.float16)  # For TI output
        
        if self.fpp.TRK_enable:
            self.fpp.TRK_update_poss_matrix(valid_points_list)
            
            obj_status_list = []
            for person in self.fpp.TRK_people_list:
                obj_cp, obj_size, obj_status = person.get_info()
                obj_status_list.append(obj_status)
                
                # Collect tracked object centroids for TI output
                if obj_status >= 0 and len(obj_cp) > 0:
                    # Create a point with [x, y, z, velocity=0, snr=status*100]
                    # Using SNR to encode status for visualization
                    tracked_point = np.array([[
                        obj_cp[0, 0], obj_cp[0, 1], obj_cp[0, 2],
                        0,  # velocity
                        (obj_status + 1) * 100  # encode status in SNR
                    ]], dtype=np.float16)
                    tracked_points = np.concatenate([tracked_points, tracked_point])
                
                if ax1 is not None:
                    self._plot(ax1, obj_cp[:, 0], obj_cp[:, 1], obj_cp[:, 2], 
                              marker='o', color=OS_colormap[obj_status])

            # Auto save
            if self.AUTOSAVE_ENABLE:
                if len(obj_status_list) > 0 and max(obj_status_list) >= 0:
                    self.autosave_flag.value = True
                else:
                    self.autosave_flag.value = False

        # >>> TI OUTPUT OPTION: tracked_obj <<<
        if self.ti_bridge and self.ti_output_stage == 'tracked_obj':
            self.ti_bridge.send_frame(tracked_points)

        # >>> TI OUTPUT OPTION: all (send clustered by default for 'all') <<<
        if self.ti_bridge and self.ti_output_stage == 'all':
            # For 'all' mode, we send the most informative data: clustered points
            # You could extend this to send multiple TLVs with different data
            if len(valid_points_total) > 0:
                self.ti_bridge.send_frame(valid_points_total)
            else:
                self.ti_bridge.send_frame(np.zeros((0, 5)))

        # ================================================================
        # End of frame
        # ================================================================
        if ax1 is not None:
            self._detect_key_press(0.001)
        else:
            time.sleep(0.001)

    def _plot(self, ax, x, y, z, fmt='', **kwargs):
        if ax is None:
            return
        if len(fmt) > 0:
            if self.dimension == '2D':
                ax.plot(x, y, fmt)
            elif self.dimension == '3D':
                ax.plot3D(x, y, z, fmt)
        else:
            for i in ['marker', 'linestyle', 'color']:
                if not (i in kwargs):
                    kwargs[i] = 'None'
            if self.dimension == '2D':
                ax.plot(x, y, marker=kwargs['marker'], linestyle=kwargs['linestyle'], color=kwargs['color'])
            elif self.dimension == '3D':
                ax.plot3D(x, y, z, marker=kwargs['marker'], linestyle=kwargs['linestyle'], color=kwargs['color'])

    def _detect_key_press(self, timeout):
        keyPressed = plt.waitforbuttonpress(timeout=timeout)
        plt.gcf().canvas.mpl_connect('key_press_event', self._press)
        if keyPressed:
            if self.the_key == 'escape':
                self.run_flag.value = False
            if self.MANSAVE_ENABLE:
                if self.the_key == '+':
                    self.mansave_flag.value = 'image'
                elif self.the_key == '0':
                    self.mansave_flag.value = 'video'

    def _press(self, event):
        self.the_key = event.key

    def _log(self, txt):
        print(f'[{self.__class__.__name__}]\t{txt}')

    def __del__(self):
        if self.fig is not None:
            plt.close(self.fig)
        if self.ti_bridge is not None:
            self.ti_bridge.stop()
        self._log(f"Closed. Timestamp: {datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
        self.status['Module_VIS'] = False
        self.run_flag.value = False


# Backwards compatibility
Visualizer = VisualizerTI