import math
import pickle

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator

from library import FramePProcessor, folder_clean_recreate, SNRV_filter, np_filter

RP_colormap = ['C5', 'C7', 'C8']  # the colormap for radar raw points
SNR_colormap = ['lavender', 'thistle', 'violet', 'darkorchid', 'indigo']  # the colormap for radar energy strength
OS_colormap = ['grey', 'green', 'gold', 'red']  # the colormap for object status


class Visualizer:
    def __init__(self, cur_dataset, interval=0.1, image_output_enable=False, **kwargs_CFG):
        """
        pass config static parameters
        """
        """ module own config """
        VIS_CFG = kwargs_CFG['VISUALIZER_CFG']
        self.dimension = VIS_CFG['dimension']
        self.VIS_xlim = VIS_CFG['VIS_xlim']
        self.VIS_ylim = VIS_CFG['VIS_ylim']
        self.VIS_zlim = VIS_CFG['VIS_zlim']

        """ other configs """
        self.RDR_CFG_LIST = kwargs_CFG['RADAR_CFG_LIST']

        """self content"""
        self.cur_dataset = cur_dataset
        self.interval = interval
        self.obj_path_saved = np.ndarray([0, 5])
        self.image_output_enable = image_output_enable
        if self.image_output_enable:
            self.image_dir = './temp/'
            folder_clean_recreate(self.image_dir)
            self.image_dpi = 150

        self.fpp = FramePProcessor(**kwargs_CFG)  # call other class

        # setup for matplotlib plot
        matplotlib.use('TkAgg')  # set matplotlib backend
        # plt.rcParams['toolbar'] = 'None'  # disable the toolbar
        # create a figure
        self.fig = plt.figure(figsize=(9, 9))
        # adjust figure position
        mngr = plt.get_current_fig_manager()
        mngr.window.wm_geometry('+150+30')
        # draws a completely frameless window
        win = plt.gcf().canvas.manager.window
        win.overrideredirect(1)
        # interactive mode on, no need plt.show()
        plt.ion()

    def run(self):
        if self.dimension == '2D':
            # create a plot
            ax1 = self.fig.add_subplot(111)

            for idx, cur_data in enumerate(self.cur_dataset):
                # clear and reset
                plt.cla()
                ax1.set_xlim(self.VIS_xlim[0], self.VIS_xlim[1])
                ax1.set_ylim(self.VIS_ylim[0], self.VIS_ylim[1])
                ax1.xaxis.set_major_locator(LinearLocator(5))  # set axis scale
                ax1.yaxis.set_major_locator(LinearLocator(5))
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')
                ax1.set_title('Radar')
                # update the canvas
                self._update_canvas(ax1, cur_data)
                print(idx)
                if self.image_output_enable:
                    self.fig.savefig(self.image_dir + f'{idx:03d}.png', dpi=self.image_dpi)

        elif self.dimension == '3D':
            # create a plot
            ax1 = self.fig.add_subplot(111, projection='3d')

            spin = 0
            ax1.view_init(ax1.elev, ax1.azim + 140)
            for idx, cur_data in enumerate(self.cur_dataset):
                # clear and reset
                plt.cla()
                ax1.set_xlim(self.VIS_xlim[0], self.VIS_xlim[1])
                ax1.set_ylim(self.VIS_ylim[0], self.VIS_ylim[1])
                ax1.set_zlim(self.VIS_zlim[0], self.VIS_zlim[1])
                ax1.xaxis.set_major_locator(LinearLocator(3))  # set axis scale
                ax1.yaxis.set_major_locator(LinearLocator(3))
                ax1.zaxis.set_major_locator(LinearLocator(3))
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')
                ax1.set_zlabel('z')
                ax1.set_title('Radar')
                spin += 0.04
                ax1.view_init(ax1.elev - 0.5 * math.sin(spin), ax1.azim - 0.3 * math.sin(1.5 * spin))  # spin the view angle
                # update the canvas
                self._update_canvas(ax1, cur_data)
                print(idx)
                if self.image_output_enable:
                    self.fig.savefig(self.image_dir + f'{idx:03d}.png', dpi=self.image_dpi)

    def _update_canvas(self, ax1, cur_data):
        # draw radar point
        for RDR_CFG in self.RDR_CFG_LIST:
            self._plot(ax1, [RDR_CFG['pos_offset'][0]], [RDR_CFG['pos_offset'][1]], [RDR_CFG['pos_offset'][2]], marker='o', color='DarkRed')

        # get values from queues of all radars
        val_data_allradar = np.ndarray([0, 5], dtype=np.float16)
        SNR_noise_allradar = np.ndarray([0, 5], dtype=np.float16)
        for i, RDR_CFG in enumerate(self.RDR_CFG_LIST):
            data_1radar = cur_data[RDR_CFG['name']]

            # apply SNR and speed filter for each radar channel
            val_data, SNR_noise = SNRV_filter(data_1radar, RDR_CFG['SNRV_threshold'])
            val_data_allradar = np.concatenate([val_data_allradar, val_data])
            SNR_noise_allradar = np.concatenate([SNR_noise_allradar, SNR_noise])

        # apply global boundary filter
        val_data_allradar = self.fpp.FPP_boundary_filter(val_data_allradar)
        SNR_noise_allradar = self.fpp.FPP_boundary_filter(SNR_noise_allradar)
        # apply global energy strength filter
        val_data_allradar, global_SNR_noise = self.fpp.FPP_SNRV_filter(val_data_allradar)
        # apply background noise filter
        val_data_allradar = self.fpp.BGN_filter(val_data_allradar)

        # val_data_allradar[:, 0] = -val_data_allradar[:, 0]
        # val_data_allradar[:, 1] = 4 - val_data_allradar[:, 1]

        # draw signal energy strength
        for i in range(len(SNR_colormap)):
            val_data_allradar_SNR, _ = np_filter(val_data_allradar, idx=4, range_lim=(i * 100, (i + 1) * 100))
            self._plot(ax1, val_data_allradar_SNR[:, 0], val_data_allradar_SNR[:, 1], val_data_allradar_SNR[:, 2], marker='.', color=SNR_colormap[i])

        # draw valid point, DBSCAN envelope
        # vertices_list, valid_points_list, _, DBS_noise = self.fpp.DBS(val_data_allradar)
        vertices_list, valid_points_list, _, DBS_noise = self.fpp.DBS_dynamic_SNR(val_data_allradar)
        for vertices in vertices_list:
            self._plot(ax1, vertices[:, 0], vertices[:, 1], vertices[:, 2], linestyle='-', color='salmon')

        # background noise filter
        if self.fpp.BGN_enable:
            # update the background noise
            if len(vertices_list) > 0:
                self.fpp.BGN_update(np.concatenate([SNR_noise_allradar, global_SNR_noise, DBS_noise]))
            else:
                self.fpp.BGN_update(np.concatenate([SNR_noise_allradar, global_SNR_noise]))
            # draw BGN area
            BGN_block_list = self.fpp.BGN_get_filter_area()
            for bgn in BGN_block_list:
                self._plot(ax1, bgn[:, 0], bgn[:, 1], bgn[:, 2], marker='.', linestyle='-', color='g')

        # tracking system
        if self.fpp.TRK_enable:
            self.fpp.TRK_update_poss_matrix(valid_points_list)
            # draw object central points
            for person in self.fpp.TRK_people_list:
                obj_cp, _, obj_status = person.get_info()
                self._plot(ax1, obj_cp[:, 0], obj_cp[:, 1], obj_cp[:, 2], marker='o', color=OS_colormap[obj_status])
                # if obj_status == 3:  # warning when object falls
                #     winsound.Beep(1000, 20)

                # if obj_cp.size > 0 and person.name == 'obj_0':
                if obj_cp.size > 0:
                    self.obj_path_saved = np.concatenate([self.obj_path_saved, np.concatenate([obj_cp, [[obj_status]], [[person.name]]], axis=1)])

        # wait at the end of each loop
        plt.pause(self.interval)

    def _plot(self, ax, x, y, z, fmt='', **kwargs):
        """
        :param ax: the current canvas
        :param x: data in x-axis
        :param y: data in y-axis
        :param z: data in z-axis
        :param fmt: plot and plot3D fmt
        :param kwargs: plot and plot3D marker, linestyle, color
        :return: None
        """
        if len(fmt) > 0:  # if fmt is using
            if self.dimension == '2D':
                # ax.plot(x, y, fmt)
                ax.plot(-np.array(x), 4-np.array(y), fmt)
            elif self.dimension == '3D':
                ax.plot3D(x, y, z, fmt)
        else:  # if para is using
            for i in ['marker', 'linestyle', 'color']:
                if not (i in kwargs):
                    kwargs[i] = 'None'
            if self.dimension == '2D':
                # ax.plot(x, y, marker=kwargs['marker'], linestyle=kwargs['linestyle'], color=kwargs['color'])
                ax.plot(-np.array(x), 4-np.array(y), marker=kwargs['marker'], linestyle=kwargs['linestyle'], color=kwargs['color'])
            elif self.dimension == '3D':
                ax.plot3D(x, y, z, marker=kwargs['marker'], linestyle=kwargs['linestyle'], color=kwargs['color'])


if __name__ == '__main__':
    from cfg.config_demo import *
    kwargs_CFG = {'VISUALIZER_CFG'          : VISUALIZER_CFG,
                  'RADAR_CFG_LIST'          : RADAR_CFG_LIST,
                  'FRAME_POST_PROCESSOR_CFG': FRAME_POST_PROCESSOR_CFG,
                  'DBSCAN_GENERATOR_CFG'    : DBSCAN_GENERATOR_CFG,
                  'BGNOISE_FILTER_CFG'      : BGNOISE_FILTER_CFG,
                  'HUMAN_TRACKING_CFG'      : HUMAN_TRACKING_CFG,
                  'HUMAN_OBJECT_CFG'        : HUMAN_OBJECT_CFG}

    # Load database
    with open('../data/XXXX', 'rb') as file:
        data = pickle.load(file)

    vis = Visualizer(data[80:], interval=0.01, image_output_enable=True, **kwargs_CFG)
    vis.run()
