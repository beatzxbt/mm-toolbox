import os
import platform
import socket
import re
import uuid
import psutil

def _get_system_info(machine: bool = False, network: bool = True, op_sys: bool = False) -> dict:
    """
    Gather basic system information about the current environment.

    Args:
        machine (bool, optional): If True, include hardware-related details like 
            architecture, processor, PID, and total RAM. Defaults to False.
        network (bool, optional): If True, include hostname, IP address, and MAC address.
            Defaults to True.
        op_sys (bool, optional): If True, include the platform version. Defaults to False.

    Returns:
        dict: A dictionary of system information. The keys included depend on which 
        flags (machine, network, op_sys) are set to True.

    Notes:
        Adapted from a StackOverflow discussion on retrieving system information in Python.
        Link: https://stackoverflow.com/questions/3103178/how-to-get-the-system-info-with-python 
    """
    info = {}

    if machine:
        info.update({
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "pid": str(os.getpid()),
            "ram": str(round(psutil.virtual_memory().total / (1024 ** 3)))
        })

    if op_sys:
        info.update({
            "platform-version": platform.version()
        })

    if network:
        info.update({
            "hostname": socket.gethostname(),
            "ip-address": socket.gethostbyname(socket.gethostname()),
            "mac-address": ':'.join(re.findall('..', '%012x' % uuid.getnode()))
        })

    return info
