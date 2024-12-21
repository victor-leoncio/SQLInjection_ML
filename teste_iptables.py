import os
import time

start = time.time()
os.system('iptables -A INPUT -s 192.168.1.100 -j DROP')
end = time.time()
print(end - start)
