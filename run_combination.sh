#!/bin/bash
exec 1>stdout.log 2>stdlog.log
path="intel/"
fd="face-detection-adas-binary-0001"
fl="landmarks-regression-retail-0009"
hp="head-pose-estimation-adas-0001"
ge="gaze-estimation-adas-0002"

source /opt/intel/openvino/bin/setupvars.sh -pyver 3.7
echo "----"
run_combination(){
    
    fd1=${path}${fd}"/"$1$"/"${fd}".xml"
    fl1=${path}${fl}"/"$2$"/"${fl}".xml"
    hp1=${path}${hp}"/"$3$"/"${hp}".xml"
    ge1=${path}${ge}"/"$4$"/"${ge}".xml"
    echo "RUNNING APP "
    echo "MODELS COMBINATION: fd, fl, hp, ge : " $1 $2 $3 $4
    echo ""
    python3.7 ./src/main.py -fd ${fd1} -fl ${fl1} -hp ${hp1} -ge ${ge1} -d CPU -i ./bin/demo.mp4 -s "all"
    echo "----"
}


run_combination "FP32-INT1" "FP32" "FP32" "FP32"
run_combination "FP32-INT1" "FP16" "FP32" "FP32"
run_combination "FP32-INT1" "FP16" "FP16" "FP32"
run_combination "FP32-INT1" "FP16" "FP16" "FP16"
run_combination "FP32-INT1" "FP32-INT8" "FP32" "FP32"
run_combination "FP32-INT1" "FP32-INT8" "FP16" "FP32"
run_combination "FP32-INT1" "FP32-INT8" "FP16" "FP16"
run_combination "FP32-INT1" "FP32-INT8" "FP32-INT8" "FP32"
run_combination "FP32-INT1" "FP32-INT8" "FP32-INT8" "FP16"
run_combination "FP32-INT1" "FP32-INT8" "FP32-INT8" "FP32-INT8"
