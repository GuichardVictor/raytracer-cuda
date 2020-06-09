import sys
from benchmark import BenchMark
from tabulate import tabulate

def main(configs):

    benchs = []
    print("=== STARTING BENCHMARK ===")
    for conf in configs:

        bench = BenchMark.new(conf)
        
        bench.setup()
        bench.run()
        bench.clean()

        benchs.append((bench, bench.exec_time))

    print('='*32, "BENCHMARK SUMMARY", '='*32)

    relative_time = benchs[0][1] # Time of first benchmark
    results = []
    for i, (bench, t) in enumerate(benchs):
        bench_id = f'({i})'
        name = bench.name
        
        cpu_usage = f'{bench.cpu_usage}%'
        ram_usage = f'{bench.memory_usage[2]}%'
        
        avg_time =  f'{t:.4f} s'

        cp_relative = (t - relative_time) / relative_time * 100
        relative =  f'{"+" if cp_relative >= 0 else ""}{cp_relative:.2f}%'

        results.append([bench_id,  name, cpu_usage, ram_usage, avg_time, relative])
    
    print(tabulate(results, headers=('id', 'Name', 'CPU load', 'RAM usage', 'Average Time', 'Relative Time (id:0)')))

if __name__ == '__main__':
    configs = sys.argv

    if len(configs) == 1:
        print('Missing config files', file=sys.stderr)
        exit(1)

    configs = configs[1:]
    main(configs)