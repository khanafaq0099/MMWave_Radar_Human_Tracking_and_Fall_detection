"""
TI Visualizer Bridge Module
===========================
This module creates a bridge between the thesis processing pipeline and TI's Industrial Visualizer.
It uses socat to create virtual serial ports and writes processed point cloud data in TI mmWave demo format.

Author: Thesis Integration
Date: 2024

Architecture:
    Thesis Pipeline → ti_visualizer_bridge → Virtual Serial Port (socat) → TI Visualizer
    
Usage:
    1. Create virtual serial ports using socat:
       $ socat -d -d pty,raw,echo=0,link=/tmp/ttyVUSB0 pty,raw,echo=0,link=/tmp/ttyVUSB1
       
    2. Configure TI Visualizer to read from /tmp/ttyVUSB1
    
    3. Run this module to write to /tmp/ttyVUSB0
"""

import struct
import serial
import time
import subprocess
import os
import threading
from datetime import datetime
from collections import deque
import numpy as np


# TI mmWave Demo Output Format Constants
MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'
HEADER_LENGTH = 40  # bytes

# TLV Types (from TI mmWave SDK)
class TLVType:
    DETECTED_POINTS = 1
    RANGE_PROFILE = 2
    NOISE_PROFILE = 3
    AZIMUTH_STATIC_HEAT_MAP = 4
    RANGE_DOPPLER_HEAT_MAP = 5
    STATS = 6
    DETECTED_POINTS_SIDE_INFO = 7
    AZIMUTH_ELEVATION_STATIC_HEAT_MAP = 8
    TEMPERATURE_STATS = 9


class TIPacketBuilder:
    """
    Builds TI mmWave demo output packets that TI Visualizer can parse.
    
    Packet Structure:
    - Magic Word (8 bytes): 0x0102030405060708 (little endian)
    - Header (32 bytes):
        - version (4 bytes)
        - totalPacketLen (4 bytes)
        - platform (4 bytes)
        - frameNumber (4 bytes)
        - timeCpuCycles (4 bytes)
        - numDetectedObj (4 bytes)
        - numTLVs (4 bytes)
        - subFrameNumber (4 bytes)
    - TLV Data (variable):
        - TLV Header (8 bytes): type (4 bytes) + length (4 bytes)
        - TLV Payload (variable)
    """
    
    def __init__(self):
        self.frame_number = 0
        self.version = 0x02010104  # SDK version mimicked
        self.platform = 0x14430114  # IWR1843 platform ID
        
    def build_packet(self, point_cloud):
        """
        Build a complete TI demo output packet from processed point cloud data.
        
        Args:
            point_cloud: numpy array of shape (N, 5) where columns are [x, y, z, velocity, snr]
            
        Returns:
            bytes: Complete packet ready to send to TI Visualizer
        """
        self.frame_number += 1
        num_points = len(point_cloud) if len(point_cloud) > 0 else 0
        
        # Build TLV for detected points
        tlv_data = b''
        if num_points > 0:
            tlv_data = self._build_detected_points_tlv(point_cloud)
            tlv_data += self._build_side_info_tlv(point_cloud)
            num_tlvs = 2
        else:
            num_tlvs = 0
        
        # Calculate total packet length
        total_length = len(MAGIC_WORD) + HEADER_LENGTH - 8 + len(tlv_data)
        
        # Build header
        header = struct.pack('<IIIIIIII',
            self.version,           # version
            total_length,           # totalPacketLen
            self.platform,          # platform
            self.frame_number,      # frameNumber
            int(time.time() * 1000) & 0xFFFFFFFF,  # timeCpuCycles
            num_points,             # numDetectedObj
            num_tlvs,               # numTLVs
            0                       # subFrameNumber
        )
        
        # Combine all parts
        packet = MAGIC_WORD + header + tlv_data
        return packet
    
    def _build_detected_points_tlv(self, point_cloud):
        """
        Build TLV for detected points in spherical coordinates.
        
        Point structure (per TI format):
        - range (float, 4 bytes)
        - azimuth (float, 4 bytes) 
        - elevation (float, 4 bytes)
        - doppler (float, 4 bytes)
        """
        points_data = b''
        
        for point in point_cloud:
            x, y, z, velocity, snr = point
            
            # Convert Cartesian to spherical coordinates
            range_val = np.sqrt(x**2 + y**2 + z**2)
            azimuth = np.arctan2(x, y)  # radians
            elevation = np.arcsin(z / max(range_val, 0.001))  # radians
            
            points_data += struct.pack('<ffff',
                float(range_val),
                float(azimuth),
                float(elevation),
                float(velocity)
            )
        
        # TLV header
        tlv_type = TLVType.DETECTED_POINTS
        tlv_length = len(points_data)
        tlv_header = struct.pack('<II', tlv_type, tlv_length)
        
        return tlv_header + points_data
    
    def _build_side_info_tlv(self, point_cloud):
        """
        Build TLV for detected points side info (SNR and noise).
        
        Side info structure:
        - snr (int16, 2 bytes)
        - noise (int16, 2 bytes)
        """
        side_info_data = b''
        
        for point in point_cloud:
            snr = int(point[4])  # SNR value
            noise = 0  # Placeholder
            side_info_data += struct.pack('<hh', snr, noise)
        
        # TLV header
        tlv_type = TLVType.DETECTED_POINTS_SIDE_INFO
        tlv_length = len(side_info_data)
        tlv_header = struct.pack('<II', tlv_type, tlv_length)
        
        return tlv_header + side_info_data


class VirtualSerialPortManager:
    """
    Manages virtual serial ports created with socat.
    Creates a pair of connected virtual serial ports.
    """
    
    def __init__(self, port_write='/tmp/ttyUSB0', port_read='/tmp/ttyUSB1'):
        self.port_write = port_write
        self.port_read = port_read
        self.socat_process = None
        self.serial_conn = None
        
    def start(self):
        """Start socat and create virtual serial port pair."""
        # Kill any existing socat processes for these ports
        try:
            subprocess.run(['pkill', '-f', f'pty.*{self.port_write}'], capture_output=True)
            time.sleep(0.5)
        except:
            pass
        
        # Remove old symlinks if they exist
        for port in [self.port_write, self.port_read]:
            try:
                os.remove(port)
            except FileNotFoundError:
                pass
        
        # Start socat to create virtual serial port pair
        cmd = [
            'socat',
            '-d', '-d',
            f'pty,raw,echo=0,link={self.port_write}',
            f'pty,raw,echo=0,link={self.port_read}'
        ]
        
        self.socat_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for ports to be created
        timeout = 5
        start_time = time.time()
        while time.time() - start_time < timeout:
            if os.path.exists(self.port_write) and os.path.exists(self.port_read):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("Failed to create virtual serial ports")
        
        time.sleep(0.5)  # Additional stabilization time
        
        # Open serial connection for writing
        self.serial_conn = serial.Serial(
            self.port_write,
            baudrate=921600,
            timeout=1
        )
        
        self._log(f"Virtual serial ports created:")
        self._log(f"  Write port (thesis → ): {self.port_write}")
        self._log(f"  Read port  (→ TI Viz): {self.port_read}")
        
    def write(self, data):
        """Write data to the virtual serial port."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.write(data)
            
    def stop(self):
        """Stop socat and cleanup."""
        if self.serial_conn:
            self.serial_conn.close()
        if self.socat_process:
            self.socat_process.terminate()
            self.socat_process.wait()
            
        # Cleanup symlinks
        for port in [self.port_write, self.port_read]:
            try:
                os.remove(port)
            except:
                pass
                
        self._log("Virtual serial ports closed")
        
    def _log(self, txt):
        print(f'[VirtualSerialPort]\t{txt}')


class TIVisualizerBridge:
    """
    Main bridge class that integrates with the thesis pipeline.
    
    Usage in thesis code:
        # In visualizer.py or main.py
        from ti_visualizer_bridge import TIVisualizerBridge
        
        # Initialize
        bridge = TIVisualizerBridge()
        bridge.start()
        
        # In the processing loop, after getting processed point cloud:
        bridge.send_frame(val_data_allradar)
        
        # On cleanup
        bridge.stop()
    """
    
    def __init__(self, port_write='/tmp/ttyUSB0', port_read='/tmp/ttyUSB1', fps=20):
        self.vspm = VirtualSerialPortManager(port_write, port_read)
        self.packet_builder = TIPacketBuilder()
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.last_frame_time = 0
        self.running = False
        self.frame_count = 0
        
    def start(self):
        """Start the bridge - creates virtual serial ports."""
        self._log("Starting TI Visualizer Bridge...")
        self.vspm.start()
        self.running = True
        self._log(f"Bridge started. Configure TI Visualizer to use: {self.vspm.port_read}")
        self._log(f"Target frame rate: {self.fps} FPS")
        
    def send_frame(self, point_cloud):
        """
        Send a processed point cloud frame to TI Visualizer.
        
        Args:
            point_cloud: numpy array of shape (N, 5) with [x, y, z, velocity, snr]
        """
        if not self.running:
            return
            
        # Rate limiting to match radar FPS
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        if elapsed < self.frame_interval:
            time.sleep(self.frame_interval - elapsed)
        
        # Ensure point_cloud is numpy array
        if not isinstance(point_cloud, np.ndarray):
            point_cloud = np.array(point_cloud)
            
        # Handle empty point cloud
        if len(point_cloud) == 0:
            point_cloud = np.zeros((0, 5))
            
        # Build and send packet
        packet = self.packet_builder.build_packet(point_cloud)
        self.vspm.write(packet)
        
        self.last_frame_time = time.time()
        self.frame_count += 1
        
        if self.frame_count % 100 == 0:
            self._log(f"Sent {self.frame_count} frames, current points: {len(point_cloud)}")
            
    def stop(self):
        """Stop the bridge and cleanup."""
        self.running = False
        self.vspm.stop()
        self._log(f"Bridge stopped. Total frames sent: {self.frame_count}")
        
    def _log(self, txt):
        print(f'[TIVisualizerBridge]\t{txt}')


# =============================================================================
# Thesis Integration Example
# =============================================================================

def integrate_with_thesis_visualizer():
    """
    Example showing how to modify visualizer.py to use TI Visualizer.
    
    Add this code to your visualizer.py _update_canvas method:
    """
    
    example_code = '''
# In visualizer.py __init__ method, add:
from ti_visualizer_bridge import TIVisualizerBridge
self.ti_bridge = TIVisualizerBridge(fps=20)
self.ti_bridge.start()

# In _update_canvas method, after processing, add:
# Send processed data to TI Visualizer
self.ti_bridge.send_frame(val_data_allradar)

# In __del__ method, add:
self.ti_bridge.stop()
'''
    return example_code


# =============================================================================
# Standalone Test Mode
# =============================================================================

def test_standalone():
    """
    Test the bridge independently with simulated data.
    """
    print("="*60)
    print("TI Visualizer Bridge - Standalone Test Mode")
    print("="*60)
    
    bridge = TIVisualizerBridge(fps=20)
    
    try:
        bridge.start()
        
        print("\n" + "="*60)
        print("Virtual serial ports are ready!")
        print(f"Configure TI Visualizer with:")
        print(f"  - Data Port: {bridge.vspm.port_read}")
        print(f"  - Baud Rate: 921600")
        print("="*60 + "\n")
        
        print("Sending simulated point cloud data...")
        print("Press Ctrl+C to stop\n")
        
        frame = 0
        while True:
            # Generate simulated point cloud data
            num_points = np.random.randint(10, 50)
            
            # Simulate human-like point cloud
            # Random points in a bounding box typical for human
            x = np.random.uniform(-0.3, 0.3, num_points)  # 60cm width
            y = np.random.uniform(1.0, 2.5, num_points)   # 1-2.5m distance
            z = np.random.uniform(0.5, 1.8, num_points)   # standing human height
            velocity = np.random.uniform(-0.5, 0.5, num_points)  # slight movement
            snr = np.random.uniform(100, 400, num_points)
            
            point_cloud = np.column_stack([x, y, z, velocity, snr])
            
            bridge.send_frame(point_cloud)
            frame += 1
            
            if frame % 20 == 0:
                print(f"Frame {frame}: {num_points} points")
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        bridge.stop()


if __name__ == '__main__':
    test_standalone()