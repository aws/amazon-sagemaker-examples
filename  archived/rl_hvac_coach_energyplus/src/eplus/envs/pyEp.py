import os
import socket
import subprocess
import sys

from eplus.envs import pyEpError


class ep_process:

    """
    Main class for pyEP representing an EnergyPlus instance

    Arguments
        ip -- the ip of the EnergyPlus server, usually 'localhost'
        port -- the port of this particular EnergyPlus instance, specified in socket.cfg. Handled automatically by socket_builder or specified manually
        building_path -- path to folder with idf, variables.cfg, and socket.cfg
        weather -- name of weather file

        Optional: eplus_path -- path to EnergyPlus version, if different from default, as specified by set_eplus_dir()

    """

    def __init__(self, ip, port, idf_file, weather, eplus_path=None):
        log_file = open("epluslog.txt", "w")

        # Not checking because it may be incorrect
        # if "BCVTB_HOME" not in os.environ:

        # assumes that bcvtb folder is in the same directory as this file
        # sets the environment variable to the current path + '/bcvtb'
        set_bcvtb_home()

        if eplus_path is None:
            global eplus_dir
            if eplus_dir is None:
                raise pyEpError.MissingEpPathError
            eplus_path = eplus_dir

        # Not sure what this was meant for
        # if not eplus_path.endswith("/time"):
        #   eplus_path = eplus_path # + "/" # there was an extra slash

        # EnergyPlus requires that the process execute in the directory
        # as the idf file and socket config file. Tested only on Mac
        idf_dir = os.path.dirname(idf_file)
        os.chdir(idf_dir)

        if os.name is "nt":  # windows
            eplus_script = eplus_path + "RunEplus"
            idf_path = building_path + "\\" + idf[:-4]
            self.p = subprocess.Popen(
                [eplus_script, idf_path, weather], stdout=log_file, shell=True, cwd=building_path
            )
        else:  # linux or mac
            eplus_script = eplus_path + "energyplus"
            idf_path = os.path.join(os.path.dirname(__file__), idf_file)
            weather_path = os.path.join(os.path.dirname(__file__), weather)
            self.p = subprocess.Popen([eplus_script, "-w", weather_path, idf_path], stdout=log_file)

        print("Using E+ executable: " + eplus_script)
        print("Using IDF file: " + idf_file)
        print("Creating E+ Process: " + eplus_script + " -w " + weather + " " + idf_path)

        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((ip, port))
        print("Started waiting for connection on %s %s" % (ip, port))
        s.listen(1)
        remote, address = s.accept()
        self.remote = remote
        print("Got connection from Host " + str(address[0]) + " Port " + str(address[1]))

    def close(self):
        print("Closing E+")
        self.write("2 1\n")
        self.remote.shutdown(socket.SHUT_RDWR)
        self.remote.close()

    def read(self):
        data = ""
        try:
            while True:
                packet = self.remote.recv(1024)
                packet = packet.decode("utf-8")
                data = data + packet
                if "\n" in packet:  # \n is end flag
                    break

        except socket.error:
            print("Socket Error")
            raise pyEpError.EpReadError

        return data

    def write(self, packet):
        # try:
        packet = packet.encode("utf-8")
        self.remote.send(packet)

    # except socket.error as err:
    #    raise pyEpError.EpWriteError

    # Takes in a packet from ep_process.read() and returns a list of lists corresponding to the real, int, and boolean values
    # Returns an empty list if there are no more outputs, or if an error occured
    def decode_packet(self, packet):
        comp = packet.split(" ")
        comp = comp[:-1]
        comp_values = [float(s) for s in comp]
        output = []
        if comp_values[0] == 2:  # Version 2
            if comp_values[1] == 0:  # Simulation still running
                num_real = int(comp_values[2])
                num_int = int(comp_values[3])
                num_bool = int(comp_values[4])
                time = comp_values[5]

                reals = comp_values[6 : 6 + num_real]
                ints = [int(comp_values[i]) for i in range(6 + num_real, 6 + num_real + num_int)]
                bools = [
                    comp_values[i] == 1
                    for i in range(6 + num_real + num_int, 6 + num_real + num_int + num_bool)
                ]
                output.append(reals)
                output.append(ints)
                output.append(bools)
            else:
                switch = {
                    1: "Simulation Finished. No output",
                    -10: "Initialization Error",
                    -20: "Time Integration Error",
                    -1: "An Unspecified Error Occured",
                }
                print(switch.get(comp_values[1]))
        else:
            raise pyEpError.VersionError
        return output

    # Takes in a list of lists with the real, int, and boolean values to input
    def encode_packet(self, setpoints, time):
        comp = [2, 0, len(setpoints[0]), len(setpoints[1]), len(setpoints[2]), time]
        for i in range(0, 3):
            comp.extend(setpoints[i])
        str_comp = [str(val) for val in comp]
        str_comp.extend("\n")
        output = " ".join(str_comp)
        return output

    # Returns a list of float outputs from E+
    def decode_packet_simple(self, packet):
        comp = packet.split(" ")
        comp = comp[:-1]
        comp_values = [float(s) for s in comp]
        output = []
        if comp_values[0] == 2:  # Version 2
            if comp_values[1] == 0:  # Simulation still running
                num_real = int(comp_values[2])
                time = comp_values[5]

                reals = comp_values[6 : 6 + num_real]
                output = reals
            else:
                switch = {
                    1: "Simulation Finished. No output",
                    -10: "Initialization Error",
                    -20: "Time Integration Error",
                    -1: "An Unspecified Error Occured",
                }
                print(switch.get(comp_values[1]))
        else:
            raise pyEpError.VersionError
        return output

    # Encodes all setpoints as reals to input to energyplus
    def encode_packet_simple(self, setpoints, time):
        comp = [2, 0, len(setpoints), 0, 0, time]
        comp.extend(setpoints)

        str_comp = [str(val) for val in comp]
        str_comp.extend("\n")
        output = " ".join(str_comp)
        return output


def set_bcvtb_home():
    path = os.path.dirname(os.path.abspath(__file__)) + "/bcvtb"
    os.environ["BCVTB_HOME"] = path  # visible in this process + all children
    print("Setting BCVTB_HOME to ", path)


def set_eplus_dir(path):
    global eplus_dir

    if path is not None:
        if not path.endswith("/"):
            path = path + "/"

    eplus_dir = path


"""
Energy Plus Protocol Version 1 & 2:
Packet has the form:
      v f dr di db t r1 r2 ... i1 i2 ... b1 b2 ...
where
  v    - version number (1,2)
  f    - flag (0: communicate, 1: finish, -10: initialization error,
               -20: time integration error, -1: unknown error)
  dr   - number of real values
  di   - number of integer values
  db   - number of boolean values
  t    - current simulation time in seconds (format %20.15e)
  r1 r2 ... are real values (format %20.15e)
  i1 i2 ... are integer values (format %d)
  b1 b2 ... are boolean values (format %d)
Note that if f is non-zero, other values after it will not be processed.
"""

if __name__ == "__main__":
    print("Do not run this file as a script. See example.py for an example of how to use pyEp")
