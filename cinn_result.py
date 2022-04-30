TRAIN_LOG = "./log/default_train.log"
MEM_LOG = "./log/default_mem_usage.log"

def get_max_mem(path):
    with open(path, 'r') as fin:
        lines = fin.readlines()

        max_mem = 0
    for i in range(1, len(lines)):
        mem = int(lines[i].split()[0])
        if mem > max_mem:
            max_mem = mem
    
    return max_mem

def get_avg_speed(path, begin_step, end_step):
    total_ips = 0.0
    n = 0
    with open(path, 'r') as fin:
        for line in fin:
            if "ppcls INFO: epoch:0" in line:
                words = line.split()
                
                step_num = int(words[6].split(":")[1])
                print(step_num)
                if step_num >= begin_step and step_num < end_step:
                    ips = float(words[22])
                    total_ips += ips
                    n += 1
    if n == 0:
        return 0, 0
    return total_ips / n, n

def main():
    speed_and_num = get_avg_speed(TRAIN_LOG, 500, 3500)
    print("Speed (ips) = " + str(speed_and_num[0]) + ", num = " + str(speed_and_num[1]))
    print("Memory (MiB) = " + str(get_max_mem(MEM_LOG)))

if __name__ == "__main__":
    main()

