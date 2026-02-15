"""
Designed for data collection from radars, abbr. RDR
"""

import multiprocessing
import re
import time
from datetime import datetime

import numpy as np
import serial
import struct
from library.TI.parser_mmw_demo import parser_one_mmw_demo_output_packet
from library.frame_early_processor import FrameEProcessor

header_length = 8 + 32
magic_word = b'\x02\x01\x04\x03\x06\x05\x08\x07'

stop_word = 'sensorStop'
HEADER_LENGTH = 40  # bytes (8 magic + 32 header info)
# TLV Types for 3D People Tracking
TLV_TYPE_TRACKER_PROC_TARGET_LIST = 1010  # 3D target list
TLV_TYPE_TARGET_HEIGHT = 1012  # Height TLV
TLV_TYPE_COMPRESSED_POINTS = 1020  # Compressed Point Cloud TLV


class RadarReader:
    def __init__(self, run_flag, radar_rd_queue, shared_param_dict, **kwargs_CFG):
        """
        get shared values and queues
        """
        self.run_flag = run_flag
        self.radar_rd_queue = radar_rd_queue

        self.status = shared_param_dict['proc_status_dict']
        self.status[kwargs_CFG['RADAR_CFG']['name']] = True

        """
        pass config static parameters
        """
        """ module own config """
        RDR_CFG = kwargs_CFG['RADAR_CFG']
        self.name = RDR_CFG['name']
        self.cfg_port_name = RDR_CFG['cfg_port_name']
        self.data_port_name = RDR_CFG['data_port_name']
        self.cfg_file_name = RDR_CFG['cfg_file_name']

        """self content"""
        self.fep = FrameEProcessor(**kwargs_CFG)  # call other class
        self.cfg_port = None
        self.data_port = None

        self._log('Start...')

    # radar connection
    def connect(self) -> bool:
        try:
            cfg_port, data_port = self._connect_port(self.cfg_port_name, self.data_port_name)
            self._send_cfg(self._read_cfg(self.cfg_file_name), cfg_port, print_enable=1)
        except:
            return False
        # set property value
        self.cfg_port = cfg_port
        self.data_port = data_port
        return True

    # module entrance
    
    def run(self):
        """Main loop - read, parse TLV 1010/1020, transform, queue"""
        if not self.connect():
            self._log(f"Radar {self.name} Connection Failed")
            self.run_flag.value = False
            return

        data_buffer = b''
        self._log('Starting data acquisition...')
        
        while self.run_flag.value:
            try:
                # Read available data
                bytes_available = self.data_port.in_waiting
                if bytes_available > 0:
                    data_buffer += self.data_port.read(bytes_available)
                
                # Look for complete frame (at least 2 magic words)
                if data_buffer.count(magic_word) >= 2:
                    # Find first magic word
                    start_idx = data_buffer.find(magic_word)
                    if start_idx == -1:
                        continue
                    
                    # Extract frame starting from magic word
                    frame_data = data_buffer[start_idx:]
                    
                    # Parse the frame to get tracks AND point cloud
                    parsed_data = self._parse_frame(frame_data)
                    
                    if parsed_data is not None:
                        point_cloud = parsed_data['point_cloud']
                        heights = parsed_data['heights']
                        num_points = parsed_data['num_points']
                        frame_length = parsed_data['frame_length']
                        try:
                            transformed_points = None
                            if point_cloud is not None and len(point_cloud) > 0:
                                transformed_points = self.fep.FEP_accumulate_update(point_cloud)
                            
                                self.radar_rd_queue.put(transformed_points)
                            
                        except Exception as e:
                            self._log(f'Transform error: {e}')
                            import traceback
                            traceback.print_exc()
                            
                        # Remove processed frame from buffer
                        if frame_length > 0:
                            data_buffer = data_buffer[start_idx + frame_length:]
                        else:
                            data_buffer = data_buffer[start_idx + len(magic_word):]
                        
                    else:
                        # Parse failed, skip magic word
                        data_buffer = data_buffer[start_idx + len(magic_word):]
                
                # Prevent buffer overflow
                if len(data_buffer) > 100000:
                    self._log(f'Warning: Buffer overflow, clearing')
                    data_buffer = b''
                    
            except Exception as e:
                self._log(f'Error in main loop: {e}')
                time.sleep(0.01)

    def _parse_frame(self, data):
        """
        Parse a single frame containing TLV 1010, 1012, and 1020
        Returns dict with all parsed data or None on error
        """
        try:
            if len(data) < HEADER_LENGTH:
                return None
            
            # Parse header (40 bytes after magic word)
            header = data[8:HEADER_LENGTH]
            
            version = struct.unpack('I', header[0:4])[0]
            total_packet_len = struct.unpack('I', header[4:8])[0]
            platform = struct.unpack('I', header[8:12])[0]
            frame_number = struct.unpack('I', header[12:16])[0]
            time_cpu_cycles = struct.unpack('I', header[16:20])[0]
            num_detected_obj = struct.unpack('I', header[20:24])[0]
            num_tlvs = struct.unpack('I', header[24:28])[0]
            subframe_number = struct.unpack('I', header[28:32])[0]
            
            if len(data) < total_packet_len:
                return None
            
            # Parse TLVs
            tlv_start = HEADER_LENGTH
            tracks = []
            heights = None
            point_cloud = None
            num_targets = 0
            num_points = 0
            
            for _ in range(num_tlvs):
                if tlv_start + 8 > len(data):
                    break
                
                tlv_type = struct.unpack('I', data[tlv_start:tlv_start+4])[0]
                tlv_length = struct.unpack('I', data[tlv_start+4:tlv_start+8])[0]
                tlv_data_start = tlv_start + 8
                # print(f'Found TLV: type={tlv_type}, length={tlv_length}')
                # TLV 1010 (3D Target List)
                # if tlv_type == TLV_TYPE_TRACKER_PROC_TARGET_LIST:
                #     tracks, num_targets = self._parse_tlv_1010_data(
                #         data[tlv_data_start:tlv_data_start+tlv_length])
                
                # TLV 1020 (Compressed Point Cloud)
                if tlv_type == TLV_TYPE_COMPRESSED_POINTS:
                    # print('Parsing TLV 1020: Compressed Point Cloud')
                    point_cloud, num_points = self._parse_tlv_1020_data(
                        data[tlv_data_start:tlv_data_start+tlv_length])
                
                # TLV 1012 (Target Height)
                # elif tlv_type == TLV_TYPE_TARGET_HEIGHT:
                #     heights = self._parse_tlv_1012_data(
                #         data[tlv_data_start:tlv_data_start+tlv_length])
                
                tlv_start = tlv_data_start + tlv_length
            
            return {
                'tracks': tracks,
                'point_cloud': point_cloud,
                'heights': heights,
                'frame_number': frame_number,
                'frame_length': total_packet_len,
                'num_targets': num_targets,
                'num_points': num_points
            }
            
        except Exception as e:
            self._log(f'Parse error: {e}')
            self.parse_errors += 1
            return None

    def _parse_tlv_1020_data(self, tlv_data):
        """
        Parse TLV 1020: Compressed Point Cloud
        
        Input format from radar (compressed spherical):
        - Header: 5 floats (elevationUnit, azimuthUnit, dopplerUnit, rangeUnit, snrUnit)
        - Per point: elevation(int8), azimuth(int8), doppler(int16), range(uint16), snr(uint16)
        
        Output: Cartesian coordinates [x, y, z, doppler, snr] for easier transformation
        
        TI's spherical to Cartesian conversion (matching visualizer):
        - X = Range * sin(Azimuth) * cos(Elevation)
        - Y = Range * cos(Azimuth) * cos(Elevation)  
        - Z = Range * sin(Elevation)
        """
        try:
            if len(tlv_data) < 20:
                return None, 0
            
            # Parse units (first 20 bytes = 5 floats)
            # Order: elevationUnit, azimuthUnit, dopplerUnit, rangeUnit, snrUnit
            pUnitStruct = '<5f'
            pUnit = struct.unpack(pUnitStruct, tlv_data[0:20])
            elevation_unit = pUnit[0]
            azimuth_unit = pUnit[1]
            doppler_unit = pUnit[2]
            range_unit = pUnit[3]
            snr_unit = pUnit[4]
            
            # Parse points (remaining bytes)
            # Format: elevation(int8), azimuth(int8), doppler(int16), range(uint16), snr(uint16)
            pointStruct = '<2bh2H'  # 2 signed bytes, 1 signed short, 2 unsigned shorts
            pointSize = struct.calcsize(pointStruct)  # = 8 bytes
            
            point_data = tlv_data[20:]
            num_points = len(point_data) // pointSize
            # print(f"num_points = {num_points}")
            if num_points == 0:
                return None, 0
            
            # Preallocate output array: [x, y, z, doppler, snr]
            points = np.zeros((num_points, 5), dtype=np.float32)
            
            for i in range(num_points):
                offset = i * pointSize
                
                try:
                    elevation_c, azimuth_c, doppler_c, range_c, snr_c = struct.unpack(
                        pointStruct, point_data[offset:offset + pointSize])
                except:
                    self._log(f'Point {i} parse failed')
                    break
                
                # Handle signed overflow (matching TI visualizer code)
                # The struct already handles this with 'b' for signed byte
                # and 'h' for signed short, but let's be explicit
                
                # Decompress using units
                elevation = elevation_c * elevation_unit  # radians
                azimuth = azimuth_c * azimuth_unit  # radians
                doppler = doppler_c * doppler_unit  # m/s
                range_m = range_c * range_unit  # meters
                snr = snr_c * snr_unit  # ratio
                
                # Convert spherical (range, azimuth, elevation) to Cartesian (x, y, z)
                # Using TI's convention (matching the visualizer):
                #   X = Range * sin(Azimuth) * cos(Elevation)
                #   Y = Range * cos(Azimuth) * cos(Elevation)
                #   Z = Range * sin(Elevation)
                cos_elev = np.cos(elevation)
                sin_elev = np.sin(elevation)
                cos_azim = np.cos(azimuth)
                sin_azim = np.sin(azimuth)
                
                x = range_m * sin_azim * cos_elev
                y = range_m * cos_azim * cos_elev
                z = range_m * sin_elev
                
                points[i, :] = [x, y, z, doppler, snr]
                # print(f'Point {i}: x={x:.2f}, y={y:.2f}, z={z:.2f}, doppler={doppler:.2f}, snr={snr:.2f}')
            
            return points, num_points
            
        except Exception as e:
            self._log(f'Error parsing TLV 1020: {e}')
            import traceback
            traceback.print_exc()
            return None, 0

    # connect the ports
    def _connect_port(self, cfg_port_name, data_port_name):
        try:
            cfg_port = serial.Serial(cfg_port_name, baudrate=115200)
            data_port = serial.Serial(data_port_name, baudrate=921600)
            assert cfg_port.is_open and data_port.is_open
            self._log('Hardware connected')
            return cfg_port, data_port
        except serial.serialutil.SerialException:
            return

    # read cfg file
    def _read_cfg(self, _cfg_file_name):
        cfg_list = []
        with open(_cfg_file_name) as f:
            lines = f.read().split('\n')
        for line in lines:
            if not (line.startswith('%') or line == ''):
                cfg_list.append(line)
        return cfg_list

    # send cfg list
    def _send_cfg(self, cfg_list, cfg_port, print_enable=1):
        for line in cfg_list:
            # send cfg line by line
            line = (line + '\n').encode()
            cfg_port.write(line)
            # wait for port response
            while cfg_port.inWaiting() <= 20:
                pass
            time.sleep(0.01)
            res_str = cfg_port.read(cfg_port.inWaiting()).decode()
            res_list = [i for i in re.split('\n|\r', res_str) if i != '']
            if print_enable == 1:
                self._log('\t'.join(res_list[-1:] + res_list[0:-1]))
        self._log('cfg SENT\n')

    # print with device name
    def _log(self, txt):
        print(f'[{self.name}]\t{txt}')

    def __del__(self):
        # stop the radar
        try:
            self._send_cfg([stop_word], self.cfg_port)
            self.cfg_port.close()
            self.data_port.close()
        except:
            pass
        self._log(f"Closed. Timestamp: {datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
        self.status[self.name] = False
        self.run_flag.value = False


if __name__ == '__main__':
    config = {'name'          : 'Test',
              'cfg_port_name' : 'COM4',
              'data_port_name': 'COM5',
              'cfg_file_name' : '../cfg/IWR1843_3D.cfg',
              'xlim'          : (-2, 2),
              'ylim'          : (0.2, 4),
              'zlim'          : (-2, 2),
              'pos_offset'    : (0, 0),  # default pos_offset is (0, 0)
              'facing_angle'  : 0,  # facing_angle is 0 degree(forward in field is up in map, degree 0-360 counted clockwise)
              'enable_save'   : True,  # data saved for single radar
              'save_length'   : 10,  # saved frame length for single radar
              }
    v = multiprocessing.Manager().Value('b', True)
    q = multiprocessing.Manager().Queue()
    r = RadarReader(v, q, config)
    success = r.connect()
    if not success:
        raise ValueError(f'Radar {config["name"]} Connection Failed')
    r.run()
