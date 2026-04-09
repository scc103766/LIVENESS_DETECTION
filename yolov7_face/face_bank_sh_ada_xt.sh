#!/bin/sh
# 2025 01 07 这个比 flask 快了一倍
# 定义设备和端口
devices="cuda:0 cuda:1 "
ports="15111 15112"

# 将设备和端口字符串转换为数组形式
device_arr=$(echo $devices)
port_arr=$(echo $ports)

# 初始化计数器
i=0

# 查找并杀掉已经运行的 face_bank_service_fast_ada.py 进程
echo "Killing existing face_bank_service_fast_ada.py processes..."
ps aux | grep 'face_bank_service_fast_ada_xt.py' | grep -v 'grep' | awk '{print $2}' | xargs kill -9
if [ $? -eq 0 ]; then
  echo "Existing processes killed."
else
  echo "No existing processes to kill."
fi

# 循环启动服务
for device in $device_arr; do
  port=$(echo $ports | cut -d' ' -f$((i + 1)))

  # 设置日志文件名称
  log_file="nohup_1755_${device}_port_${port}.out"

  # 启动服务并将输出重定向到对应的日志文件
  echo "Starting service on device $device and port $port..."

  # 启动 Flask 服务，传递 device 和 port 参数
  nohup python3 face_bank_service_fast_ada_xt.py --device $device --port $port > $log_file 2>&1 &

  # 独立子进程，避免被主脚本信号干扰
  disown

  # 输出状态
  echo "Service started on device $device and port $port, logging to $log_file"

  # 增加计数器
  i=$((i + 1))
done
