class Writer:
    def __init__(self, verbose: bool, output_file: str | None):
        """
        A utility class to write text.

        :param verbose: Print extra information, useful for debugging
        :param output_file: Print to a file. Pass None for command line output
        """
        self.verbose: bool = verbose
        if output_file == None:
            self.output = lambda line: print(line)
        else:
            self.output = lambda line: Writer._write_to_file(line)

    def write_line_verbose(self, line: str) -> None:
        """
        Write a line only when verbose is enabled.
        """
        if self.verbose:
            self.write_line(line)

    def write_line(self, line: str) -> None:
        """
        Write a line of text.
        """
        self.output(line)

    def _write_to_file(line: str, path: str) -> None:
        """
        Write a line to a file as an append.
        """
        with open(path, "a") as file:
            file.write("appended text")
