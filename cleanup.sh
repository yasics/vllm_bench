
# 递归获取所有子孙进程
function get_child_pids() {
    local parent=$1
    local children=$(pgrep -P $parent)
    for child in $children; do
        printf "$child "
        get_child_pids $child
    done
}

# 排除ixsys进程
function remove_ixsy_process() {
    local pid_list=$1
    ps -p ${pid_list} -o pid,cmd | tail -n +2 |grep -iv "^[[:space:]]*[0-9]*[[:space:]]*ixsys"|awk '{print $1}'
}

# 获取排除掉ixsys进程后的所有进程id
function get_noixsys_child_pids() {
    declare -a child_pids=($(get_child_pids $1))
    # echo ${child_pids[@]}

    child_pids=$(IFS=,; echo "${child_pids[*]}")
    # echo $child_pids

    declare -a noixsys_child_pids=($(remove_ixsy_process $(IFS=,; echo "${child_pids[*]}")))
    echo ${noixsys_child_pids[@]}
}

# 清理函数
function cleanup() {
    # 临时屏蔽所有信号，防止cleanup过程被中断
    trap '' SIGTERM SIGINT SIGHUP

    echo "开始清理进程 ${pid_list[@]} ..."
    
    # 遍历所有主进程
    for pid in "${pid_list[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            # 获取所有子进程PID
            child_pids=$(get_noixsys_child_pids $pid)
            # 如果是使用ixsys进行profiling的模式，则仅ixsys以外的子进程
            if [[ "${PROFILING}" == 1 ]]; then
                all_pids="$child_pids"
            else
                all_pids="$pid $child_pids"
            fi

            # 首先尝试正常终止
            echo "正在终止进程 $pid 及其子进程..."
            kill -TERM $all_pids 2>/dev/null

            # 等待最多5秒
            for i in {1..10}; do
                still_alive=0
                for check_pid in $all_pids; do
                    if kill -0 $check_pid 2>/dev/null; then
                        still_alive=1
                        break
                    fi
                done
                
                if [ $still_alive -eq 0 ]; then
                    break
                fi
                sleep 1
            done

            # 如果仍然存在进程，强制终止
            for check_pid in $all_pids; do
                if kill -0 $check_pid 2>/dev/null; then
                    echo "强制终止进程 $check_pid"
                    kill -9 $check_pid 2>/dev/null
                fi
            done
        fi
    done

    # 最后验证所有进程确实都已终止
    count=300
    for pid in "${pid_list[@]}"; do
        while kill -0 $pid 2>/dev/null; do
            if [[ "$count" -lt 0 ]]; then
                echo "清理失败，等待超过30s，依旧存在残留进程： ${pid} ..."
                return 1
            fi
            count=$((count-1))
            sleep 0.1
        done
    done

    echo "清理完成"
    return 0
}

function exit_with_cleanup() {
    cleanup
    exit $?
}

# 注册信号处理
trap exit_with_cleanup SIGTERM SIGINT SIGHUP