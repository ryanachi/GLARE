"""
Attack Logs to file
========================
"""

from textattack.loggers.file_logger import FileLogger


class FileLoggerSlim(FileLogger):
    """Logs the results of an attack to a file, or `stdout`."""

    def log_attack_result(self, result):
        self.num_results += 1
        self.fout.write(str(self.num_results) + "\n")
        self.fout.write(result.goal_function_result_str("ansi" if self.stdout else "file") + '\n')
        self.fout.write(result.original_text() + '\n')
        self.fout.write(result.perturbed_text() + '\n')
        self.fout.write("\n")

    def log_sep(self):
        pass

    def flush(self):
        self.fout.flush()