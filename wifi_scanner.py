import subprocess  # Allows execution of system commands
import re          # Provides regular expression matching operations
import time        # Time-related functions
import argparse    # Parses command-line arguments
import csv         # CSV file reading and writing
from datetime import datetime  # Date and time operations
import os          # Miscellaneous operating system interfaces
import logging     # Logging facility for tracking events

def scan_wifi(interface='wlan0'):
    """
    Scans for available Wi-Fi networks and returns a list of dictionaries
    containing BSSID, SSID, Frequency, RSSI, and Quality for each network.
    """
    try:
        # Execute the iwlist scan command to get Wi-Fi network information
        scan_output = subprocess.check_output(['sudo', 'iwlist', interface, 'scan'], stderr=subprocess.STDOUT)
        scan_output = scan_output.decode('utf-8')  # Decode the byte output to string
    except subprocess.CalledProcessError as e:
        # Log an error if the scan command fails
        logging.error(f"Error scanning Wi-Fi networks: {e.output.decode('utf-8')}")
        return []

    # Split the output into blocks for each cell (network)
    cells = scan_output.split('Cell ')
    networks = []

    for cell in cells[1:]:  # The first split is irrelevant
        network = {}
        
        # Extract BSSID using regex
        bssid_match = re.search(r'Address: ([\dA-Fa-f:]{17})', cell)
        network['BSSID'] = bssid_match.group(1) if bssid_match else 'N/A'
        
        # Extract SSID using regex
        ssid_match = re.search(r'ESSID:"(.*)"', cell)
        network['SSID'] = ssid_match.group(1) if ssid_match else 'Hidden'
        
        # Extract Frequency using regex
        freq_match = re.search(r'Frequency:(\d+\.\d+) GHz', cell)
        network['Frequency (GHz)'] = freq_match.group(1) if freq_match else 'N/A'
        
        # Extract RSSI (Signal level) using regex
        rssi_match = re.search(r'Signal level=([-0-9]+) dBm', cell)
        network['RSSI (dBm)'] = rssi_match.group(1) if rssi_match else 'N/A'
        
        # Extract Quality using regex
        quality_match = re.search(r'Quality=(\d+)/(\d+)', cell)
        if quality_match:
            network['Quality'] = f"{quality_match.group(1)}/{quality_match.group(2)}"
        else:
            network['Quality'] = 'N/A'
        
        # Add the extracted network information to the list
        networks.append(network)
    
    return networks

def parse_arguments():
    """
    Parses command-line arguments to determine which fields to include and scan parameters.
    """
    parser = argparse.ArgumentParser(description='Continuous Wi-Fi Scanner for Raspberry Pi')
    parser.add_argument('--interface', type=str, default='wlan0',
                        help='Wireless interface to use (default: wlan0)')
    parser.add_argument('--interval', type=int, default=5,
                        help='Scan interval in seconds (default: 5)')
    # Replaced location_id with location_X and location_Y
    parser.add_argument('--location_X', type=float, required=True,
                        help='X coordinate for the scanning location')
    parser.add_argument('--location_Y', type=float, required=True,
                        help='Y coordinate for the scanning location')
    parser.add_argument('--duration', type=int, required=True,
                        help='Total duration to perform scans (in seconds)')
    parser.add_argument('--output', type=str, default='wifi_scan_results.csv',
                        help='Output CSV file name (default: wifi_scan_results.csv)')
    return parser.parse_args()

def initialize_csv(file_name, headers):
    """
    Initializes the CSV file with headers.
    If the file already exists, it will not overwrite and will append data.
    """
    file_exists = os.path.isfile(file_name)
    if not file_exists:
        try:
            with open(file_name, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                print(f"CSV file '{file_name}' created with headers.")
        except Exception as e:
            # Log an error if the CSV file cannot be created
            logging.error(f"Failed to create CSV file '{file_name}': {e}")
            print(f"Failed to create CSV file '{file_name}': {e}")
    else:
        # Inform the user that data will be appended to existing file
        print(f"CSV file '{file_name}' already exists. Appending data.")

def setup_logging():
    """
    Sets up logging to record errors and important events.
    """
    logging.basicConfig(
        filename='wifi_scanner.log',
        level=logging.ERROR,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )

def main():
    setup_logging()  # Initialize logging
    args = parse_arguments()  # Parse command-line arguments
    interface = args.interface
    scan_interval = args.interval
    location_X = args.location_X
    location_Y = args.location_Y
    duration = args.duration
    output_file = args.output

    # Define headers based on options (replaced Location_ID with Location_X and Location_Y)
    headers = ['Timestamp', 'Location_X', 'Location_Y', 'BSSID', 'SSID', 'Frequency (GHz)', 'RSSI (dBm)', 'Quality']

    # Initialize CSV file
    initialize_csv(output_file, headers)

    # Calculate the end time based on duration
    end_time = time.time() + duration

    print(f"Starting Wi-Fi scan on interface '{interface}' every {scan_interval} seconds for {duration} seconds.")
    print(f"Results will be saved to '{output_file}'.")
    print(f"Scanning Location: X = {location_X}, Y = {location_Y}")

    try:
        while time.time() < end_time:
            networks = scan_wifi(interface)  # Perform Wi-Fi scan
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Current timestamp
            if networks:
                # Open CSV file in append mode to add scan results
                with open(output_file, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=headers)
                    for net in networks:
                        # Prepare row data for each network
                        row = {
                            'Timestamp': timestamp,
                            'Location_X': location_X,
                            'Location_Y': location_Y,
                            'BSSID': net.get('BSSID', 'N/A'),
                            'SSID': net.get('SSID', 'Hidden'),
                            'Frequency (GHz)': net.get('Frequency (GHz)', 'N/A'),
                            'RSSI (dBm)': net.get('RSSI (dBm)', 'N/A'),
                            'Quality': net.get('Quality', 'N/A')
                        }
                        writer.writerow(row)  # Write row to CSV
                print(f"[{timestamp}] Scan completed. {len(networks)} networks found.")
            else:
                # Inform the user if no networks were found or an error occurred
                print(f"[{timestamp}] No networks found or an error occurred.")
            
            time.sleep(scan_interval)  # Wait for the next scan interval
    except KeyboardInterrupt:
        # Handle user interruption gracefully
        print("\nWi-Fi scanning stopped by user.")
    except Exception as e:
        # Log any unexpected errors
        logging.error("An unexpected error occurred", exc_info=True)
        print(f"An unexpected error occurred: {e}")
    finally:
        # Final message upon completion
        print(f"Scanning session ended. Results saved to '{output_file}'.")

if __name__ == "__main__":
    main()
