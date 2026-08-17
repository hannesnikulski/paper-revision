import argparse
import logging
import multiprocessing as mp
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from pysr import PySRRegressor, TemplateExpressionSpec, TensorBoardLoggerSpec


# ----------------------------------------------------------------------------------------------------
# Argument Parser
# ----------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------
    parser.add_argument(
        "--logfile",
        default=None,
        help="Specifies the output file for the Python logger. STDOUT is used as the default."
    )
    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
        help="Set the logging level for the Python logger."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="What verbosity level to use."
    )
    # --------------------------------------------------
    # TensorBoard logging
    # --------------------------------------------------
    parser.add_argument(
        "--logging",
        action="store_true",
        help="Enables TensorBoard logging."
    )
    # --------------------------------------------------
    # Paths
    # --------------------------------------------------
    parser.add_argument(
        "--datapath",
        type=Path,
        help="Directory of the curve data",
    )
    parser.add_argument(
        "--loss-function",
        type=Path,
        help="Path to the loss function",
    )
    # --------------------------------------------------
    # SLURM cluster manager
    # --------------------------------------------------
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Enables PySR to use the SLURM cluster manager."
    )

    return parser


# ----------------------------------------------------------------------------------------------------
# Logger
# ----------------------------------------------------------------------------------------------------
def configure_logger(args: argparse.Namespace) -> logging.Logger:
    logging.basicConfig(
        filename=args.logfile,
        filemode="a",
        format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, args.loglevel)
    )

    logger = logging.getLogger("PySR Search")
    return logger


def main():
    parser = build_parser()
    args = parser.parse_args()

    logger = configure_logger(args)
    logger.info("Start new PySR search")

    # ----------------------------------------------------------------------------------------------------
    # Number of processes
    # ----------------------------------------------------------------------------------------------------
    NPROCS = int(os.environ.get('SLURM_NPROCS', 0))
    if args.slurm:
        if NPROCS < 1:
            logger.warning('Unable to read `SLURM_NPROCS` environment variable!')
            quit(-1)

    else:
        if NPROCS >= 1:
            logger.info('SLURM cluster manager detected. Run this file again with the `--slurm` flag!')
            quit(-1)

        logger.info('Running in multithreaded mode.')
        NPROCS = mp.cpu_count()

    logger.info(f'Number of processes / threads: {NPROCS}.')

    # ----------------------------------------------------------------------------------------------------
    # Config
    # ----------------------------------------------------------------------------------------------------
    pysrConfig = {
        # Search space
        'maxsize': 22,

        # Search size
        'niterations': 1_000,
        'populations': 3 * NPROCS,

        # Mutations
        # 'weight_optimize': 0.01,
        # 'skip_mutation_failures': False,

        # Constant optimization
        'optimizer_iterations': 16,
        'optimize_probability': 0.5,

        # Stopping criteria
        'early_stop_condition': 'f(loss, complexity) = (loss < 1e-9) && (complexity < 18)',

        # Performance
        'precision': 64,
        'batching': True,
        'fast_cycle': True,

        'turbo': True,
        'bumper': True,

        # Monitoring
        'verbosity': args.verbose,

        # --------------------------------------------------
        # Operators and constraints
        # --------------------------------------------------
        'binary_operators': [
            '+',
            '*',
            '/',
            '^'
        ],
        'unary_operators': [
            'square',
            'sqrt',
            'exp', 'log',
            'tanh'
        ],
        'nested_constraints': {
            'square': {'sqrt': 0, 'square': 0, '+': 2, '*': 2},
            'exp': {'log': 0, 'exp': 0, '+': 2, '*': 2},
            'log': {'exp': 0, 'log': 0, '+': 2, '*': 2},
            '+': {'+': 2, '*': 2},
            '*': {'*': 2, '+': 2},
            '/': {'/': 0, '+': 2, '*': 2},
            '^': {'^': 0, '+': 2, '*': 2},
            'sqrt': {'sqrt': 0, '+': 2, '*': 2},
            'tanh': {'tanh': 0, '+': 2, '*': 2}
        },
        'constraints': {'/': (-1, 5), '^': (-1, 1)},
        'complexity_of_constants': 10,
    }

    # --------------------------------------------------
    # Output Directory
    # --------------------------------------------------
    outputDirectory = Path(__file__).parent / 'output'
    runId = datetime.now().strftime(f'Run_%Y%m%d_%H%M')
    runDirectory = outputDirectory / runId

    pysrConfig.update({
        'run_id': runId,
        'output_directory': str(outputDirectory),
    })

    logger.info(f'Output: {runDirectory}')

    # --------------------------------------------------
    # TensorBoard Logging
    # --------------------------------------------------
    if args.logging:
        logger_spec = TensorBoardLoggerSpec(
            log_dir=str(outputDirectory / 'logs' / runId),
            # Log every 100 iterations
            log_interval=100,
        )
        pysrConfig.update({'logger_spec': logger_spec})

        logger.info('TensorBoardLogger enabled.')

    # --------------------------------------------------
    # Run in distributed mode
    # --------------------------------------------------
    if args.slurm:
        pysrConfig.update({
            'cluster_manager': 'slurm',
            'procs': NPROCS,
            'ncycles_per_iteration': 10_000,
            'parallelism': '-',
        })

        logger.info('SLURM cluster manager enabled.')

    # --------------------------------------------------
    # Use a custom loss function
    # --------------------------------------------------
    if args.loss_function:
        with open(args.loss_function, 'r') as file:
            pysrConfig.update({'loss_function_expression': file.read()})

        logger.info(f'Using custom loss function: {args.loss_function}.')

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    matrix = np.load(args.datapath)
    numCategories = np.unique(matrix[:, 2]).shape[0]

    X_with_category = matrix[:, [0, 2]]
    Y = matrix[:, [1]]

    # --------------------------------------------------
    # Create PySR model
    # --------------------------------------------------
    model = PySRRegressor(
        expression_spec=TemplateExpressionSpec(
            expressions=['f'],
            variable_names=['x', 'category'],
            parameters={
                'p1': numCategories,
                'p2': numCategories,
                'p3': numCategories,
                'p4': numCategories,
                'p5': numCategories,
            },
            # combine='p1[category] + p2[category] * x + f(x, p1[category], p2[category], p3[category], p4[category], p5[category])'
            combine='f(x, p1[category], p2[category], p3[category], p4[category], p5[category])'
        ),
        **pysrConfig
    )

    # ----------------------------------------------------------------------------------------------------
    # Save configuration
    # ----------------------------------------------------------------------------------------------------
    runDirectory.mkdir(parents=True, exist_ok=True)
    shutil.copy(__file__, runDirectory / 'source.txt')

    # ----------------------------------------------------------------------------------------------------
    # Fit
    # ----------------------------------------------------------------------------------------------------
    model.fit(X_with_category, Y)
    logger.info('PySR search completed')


if __name__ == '__main__':
    main()
