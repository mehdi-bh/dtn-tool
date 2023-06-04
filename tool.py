import sys
import os
import subprocess
import shlex
import time
import math
from io import StringIO
import psutil
import select
from PyQt5.QtGui import QTextCursor, QColor
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, QTabWidget,
                             QComboBox, QPushButton, QFileDialog, QLineEdit, QSizePolicy,
                             QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView, QHBoxLayout, QSplitter)
from PyQt5.QtCore import Qt, QIODevice, QTextStream, pyqtSignal, QThread, QTimer
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg

Solvers = {
    "cp": {
        "chuffed": {"cmd": "minizinc --solver Chuffed --all-solutions -"},
        "gecode": {"cmd": "minizinc --solver Gecode --all-solutions -"},
        "or-tools": {"cmd": "minizinc --solver OR-tools --all-solutions -"},
        "minicp": {"cmd": "java -jar minicp_solver.jar {instance} {relaxation} {variable} {value}"}
    }
}

class MiniZincGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        self.create_widgets()
        self.connect_signals()
        self.add_widgets_to_left_layout(left_layout)
        self.add_widgets_to_right_layout(right_layout)

        main_splitter = QSplitter(Qt.Horizontal)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        right_widget.setMinimumWidth(600)
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)

        main_layout.addWidget(main_splitter)

        self.setLayout(main_layout)
        self.setWindowTitle("CP Solver")
        self.show()

        sys.stdout = EmittingStream(text_written=self.write_to_console)

        self.file_path = "No file selected"

    def create_widgets(self):
        self.file_label = QLabel("Select File:")
        self.file_button = QPushButton("Choose File")
        self.filename_label = QLabel("No file selected")
        self.resolution_label = QLabel("Solver strategy:")
        self.resolution_dropdown = QComboBox()
        self.resolution_dropdown.addItems(["Satisfy (Hard Constraints)",
                                           "Minimize (Soft Constraints)"])
        self.resolution_dropdown.setEnabled(False)
        self.variable_heuristic_label = QLabel("Variable Heuristic:")
        self.value_heuristic_label = QLabel("Value Heuristic:")
        self.variable_heuristic_dropdown = QComboBox()
        self.value_heuristic_dropdown = QComboBox()
        self.variable_heuristic_dropdown.addItems(["smallest", "largest", "first_fail", "anti_first_fail", "most_constrained", "occurrence", "input_order", "max_regret","impact", "dom_w_deg"])
        self.value_heuristic_dropdown.addItems(["indomain", "indomain_min", "indomain_max", "indomain_interval", "indomain_reverse_split", "indomain_split", "indomain_split_random", "outdomain_max", "outdomain_median", "outdomain_min", "outdomain_random"])
        self.variable_heuristic_dropdown.setEnabled(False)
        self.value_heuristic_dropdown.setEnabled(False)
        self.solver_label = QLabel("Solver:")
        self.solver_dropdown = QComboBox()
        self.solver_dropdown.addItems(["chuffed", "gecode", "or-tools", "minicp"])
        self.solver_dropdown.setEnabled(False)
        self.timeout_label = QLabel("Timeout (leave blank for no timeout):")
        self.timeout_input = QLineEdit()
        self.run_button = QPushButton("Run")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.clear_console_button = QPushButton("Clear Console")
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.total_violations_label = QLabel("Total violations: 0")              
        self.constraints_table = QTableWidget(0, 1)
        self.constraints_table.horizontalHeader().setVisible(False)
        self.constraints_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.constraints_table.verticalHeader().setVisible(False)
        self.constraints_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.plot_widget = pg.PlotWidget()
        self.plot_data = self.plot_widget.plot([], [])

        self.execution_timer = QTimer()
        self.current_time = QLabel("Current time: 0")

        self.state = QLabel("State: Sleeping")

        self.tab_widget = self.create_tab_widget()
        self.clear_tabs_button = QPushButton("Clear")

    def create_tab_widget(self):
        tab_widget = QTabWidget()
        tab_widget.addTab(self.create_constraints_tab(), "Constraints")
        tab_widget.addTab(self.create_plot_tab(), "Plot")
        return tab_widget
    
    def create_constraints_tab(self):
        widget = QWidget()  
        layout = QVBoxLayout()
        layout.addWidget(self.constraints_table)
        widget.setLayout(layout) 
        return widget
    
    def create_plot_tab(self):
        widget = QWidget()  
        layout = QVBoxLayout()
        layout.addWidget(self.plot_widget)
        widget.setLayout(layout) 

        self.plot_widget.setLabel('left', 'Total Violations')
        self.plot_widget.setLabel('bottom', 'Elapsed Time', units='s')

        return widget 

    def connect_signals(self):
        self.file_button.clicked.connect(self.choose_file)
        self.run_button.clicked.connect(self.run_minizinc)
        self.stop_button.clicked.connect(self.stop_minizinc)
        self.clear_console_button.clicked.connect(self.clear_console)
        self.clear_tabs_button.clicked.connect(self.clear_tabs)
        self.solver_dropdown.currentIndexChanged.connect(self.on_solver_changed)
        self.resolution_dropdown.currentIndexChanged.connect(self.on_resolution_changed)

        self.execution_timer.timeout.connect(self.update_elapsed_time)

    def on_solver_changed(self, index):
        if self.solver_dropdown.currentText() == "minicp":
            self.resolution_dropdown.setEnabled(False)
            self.variable_heuristic_dropdown.clear()
            self.value_heuristic_dropdown.clear()
            self.resolution_dropdown.setCurrentIndex(1)
            self.variable_heuristic_dropdown.addItems(["first_fail", "max_regret", "most_constrained"])
            self.value_heuristic_dropdown.addItems(["indomain_min"])
        else:
            self.resolution_dropdown.setEnabled(True)
            self.variable_heuristic_dropdown.clear()
            self.value_heuristic_dropdown.clear()
            self.variable_heuristic_dropdown.addItems(["smallest", "largest", "first_fail", "anti_first_fail", "most_constrained", "occurrence", "input_order", "max_regret","impact", "dom_w_deg"])
            self.value_heuristic_dropdown.addItems(["indomain", "indomain_min", "indomain_max", "indomain_interval", "indomain_reverse_split", "indomain_split", "indomain_split_random", "outdomain_max", "outdomain_median", "outdomain_min", "outdomain_random"])

    def on_resolution_changed(self, index):
        if self.resolution_dropdown.currentIndex() == 0:
                self.variable_heuristic_dropdown.setEnabled(False)
                self.value_heuristic_dropdown.setEnabled(False)
        else:
            self.variable_heuristic_dropdown.setEnabled(True)
            self.value_heuristic_dropdown.setEnabled(True)

    def add_widgets_to_left_layout(self, layout):
        layout.addWidget(self.file_label)
        layout.addWidget(self.file_button)
        layout.addWidget(self.filename_label)
        layout.addWidget(self.solver_label)
        layout.addWidget(self.solver_dropdown)
        layout.addWidget(self.resolution_label)
        layout.addWidget(self.resolution_dropdown)
        layout.addWidget(self.variable_heuristic_label)
        layout.addWidget(self.variable_heuristic_dropdown)
        layout.addWidget(self.value_heuristic_label)
        layout.addWidget(self.value_heuristic_dropdown)
        layout.addWidget(self.timeout_label)
        layout.addWidget(self.timeout_input)

        # Create a QHBoxLayout for the run and stop buttons
        buttons_layout = QHBoxLayout()

        # Add the run and stop buttons to the buttons_layout
        buttons_layout.addWidget(self.run_button)
        buttons_layout.addWidget(self.stop_button)

        # Add the buttons_layout to the main layout
        layout.addLayout(buttons_layout)
        
        layout.addWidget(self.console_output)
        layout.addWidget(self.clear_console_button)
        
    def add_widgets_to_right_layout(self, layout):
        top_info_layout = QHBoxLayout()
        top_info_layout.addWidget(self.total_violations_label)
        top_info_layout.addWidget(self.current_time)
        top_info_layout.addWidget(self.state)

        layout.addLayout(top_info_layout)
        layout.addWidget(self.tab_widget)
        layout.addWidget(self.clear_tabs_button)

    def choose_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_dialog = QFileDialog(self, "Select MiniZinc File", "", "All Files (*);;MiniZinc Files (*.mzn)")
        file_dialog.setOptions(options)
        
        if file_dialog.exec_() == QFileDialog.Accepted:
            self.file_path = file_dialog.selectedFiles()[0]
            self.filename_label.setText(os.path.basename(self.file_path))

            # if file_path is minizinc file -> 
            if self.file_path.endswith(".mzn"):
                self.parse_and_display_constraints(self.file_path)
                self.solver_dropdown.clear()
                self.solver_dropdown.addItems(["chuffed", "gecode", "or-tools"])
                self.solver_dropdown.setEnabled(True)
                self.resolution_dropdown.setEnabled(True)
                self.variable_heuristic_dropdown.setEnabled(True)
                self.value_heuristic_dropdown.setEnabled(True)
            else:
                t, c, h = self.LoadDtp(self.file_path)
                content = self.MiniZincNaiveEncoder(t, c, h)
                self.writeToFile("temp.mzn", content)
                self.smt_file = str(self.file_path)
                self.file_path = "temp.mzn"
                self.parse_and_display_constraints("temp.mzn")
                self.solver_dropdown.clear()
                self.solver_dropdown.addItems(["chuffed", "gecode", "or-tools", "minicp"])
                self.solver_dropdown.setEnabled(True)
                self.resolution_dropdown.setEnabled(True)
                self.variable_heuristic_dropdown.setEnabled(True)
                self.value_heuristic_dropdown.setEnabled(True)
        else:
            self.filename_label.setText("No file selected")
            self.solver_dropdown.setEnabled(False)
            self.resolution_dropdown.setEnabled(False)
            self.variable_heuristic_dropdown.setEnabled(False)
            self.value_heuristic_dropdown.setEnabled(False)

    def LoadDtp(self, filename):
        TimePoints  = []
        Constraints = []
        W = 1

        with open(filename,"r") as tnfile:
            f = StringIO(tnfile.read())

        [N, M] = f.readline().strip().split()
        N = int(N)
        M = int(M)
        for i in range(0,N):
            [X, T] = f.readline().strip().split()
            assert (T == "c")
            TimePoints.append(X)
        for i in range(0,M):
            line = f.readline().strip().split()
            D = int(line[0])
            Constraints.append([])
            assert (line[1] == "f")
            for j in range(1,D+1):
                # 4*(j-1) + 2 = 4*j - 2
                # 4*(j-1) + 6 = 4*j + 2
                (X,Y,l,u) = tuple(line[4*j-2: 4*j+2])
                if l == "-inf":
                    l = -math.inf
                else:
                    l = int(l)
                    W = max(W, abs(l))
                if u == "+inf":
                    u = math.inf
                else:
                    u = int(u)
                    W = max(W, abs(u))
                Constraints[-1].append((Y,X,l,u))
                assert l <= u
            assert len(Constraints[-1]) == D

        assert len(TimePoints)  == N
        assert len(Constraints) == M

        return TimePoints, Constraints, W*N

    def MiniZincNaiveEncoder(self, TimePoints, Constraints, H):
        mzn = StringIO()
        NameMap = dict()

        for X in TimePoints:
            NameMap[X] = len(NameMap)+1

        mzn.write("array[1..{}] of var 0..{}: X;\n".format(len(NameMap), H))

        for disj in Constraints:
            atoms = []
            for (Y,X,l,u) in disj:
                assert l != -math.inf or u != math.inf
                tmp = []
                if l != -math.inf:
                    tmp.append("X[{Y}] - X[{X}] >= {l}".format(Y=NameMap[Y],X=NameMap[X],l=l))
                if u != math.inf:
                    tmp.append("X[{Y}] - X[{X}] <= {u}".format(Y=NameMap[Y],X=NameMap[X],u=u))
                atoms.append("({})".format(" /\ ".join(tmp)))
            mzn.write("constraint {};\n".format(" \/ ".join(atoms)))

        mzn.write("output[\"(sat)\"];\n")
        mzn.write("solve satisfy;")

        return mzn.getvalue()

    def clear_console(self):
        self.console_output.clear()
    
    def clear_tabs(self):
        # Clear the table
        for i in range(self.constraints_table.rowCount()):
            for j in range(self.constraints_table.columnCount()):
                item = self.constraints_table.item(i, j)
                if item is not None:
                    item.setBackground(QColor(255, 255, 255))  # Set the background color to white

        # Clear the plot
        self.violation_data = []
        self.time_data = []
        self.plot_data.setData(self.time_data, self.violation_data)

        # Clear the labels
        self.total_violations_label.setText("Total violations: 0")
        self.current_time.setText("Current time: 0")

    def run_minizinc(self):
        if self.file_path == "No file selected":
            print("No MiniZinc file selected")
            return

        instance = self.get_instance()

        solver_name = self.solver_dropdown.currentText()
        timeout_str = self.timeout_input.text()
        timeout = int(timeout_str) if timeout_str else None

        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.clear_tabs()
        self.console_output.clear()
        self.violation_data = []
        self.time_data = []
        self.start_time = time.time()
        self.execution_timer.start(10)
        self.state.setText("State: Running")

        if solver_name == "minicp":
            self.solver_thread = SolverThread(instance, solver_name, self.smt_file, self.variable_heuristic_dropdown.currentText().upper(), self.value_heuristic_dropdown.currentText().upper(), timeout)
        else:
            self.solver_thread = SolverThread(instance, solver_name, self.file_path, self.variable_heuristic_dropdown.currentText().upper(), self.value_heuristic_dropdown.currentText().upper(), timeout)
        self.solver_thread.finished.connect(self.solver_finished)
        self.solver_thread.output_received.connect(self.write_to_console)
        self.solver_thread.start()

    def stop_minizinc(self):
        if self.solver_thread.isRunning():
            self.solver_thread.answer = "Stopped"
            elapsed_time = time.time() - self.solver_thread.start_time
            self.solver_thread.solving_time = elapsed_time
            if self.solver_thread.process:
                parent = psutil.Process(self.solver_thread.process.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
            self.solver_thread.terminate()
            self.solver_thread.wait()

        self.execution_timer.stop()
        self.state.setText("State: Sleeping")
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_elapsed_time(self):
        timeout = int(self.timeout_input.text()) if self.timeout_input.text() else None

        elapsed_time = time.time() - self.start_time
        if timeout is None or timeout >= elapsed_time:
            self.current_time.setText(f"Elapsed time: {elapsed_time:.2f} seconds")

    def get_instance(self): 
        if self.resolution_dropdown.currentIndex() == 0:
            with open(self.file_path, 'r') as file:
                instance = file.read()
        elif self.resolution_dropdown.currentIndex() == 1:
            instance = self.relax_instance(self.file_path, self.variable_heuristic_dropdown.currentText(), self.value_heuristic_dropdown.currentText())
        return instance

    def relax_instance(self, input_file, variable_heuristic, value_heuristic):
        with open(input_file, 'r') as f:
            lines = f.readlines()

        constraint_count = sum(1 for line in lines if line.startswith('constraint'))

        new_lines = [f'array[1..{constraint_count}] of var 0..1: V;\n']

        v_index = 1
        for line in lines:
            if line.startswith('constraint'):
                new_line = f'constraint V[{v_index}] = 1 - ({line.strip()[11:].rstrip(";")});\n'
                new_lines.append(new_line)
                v_index += 1
            else:
                if not line.startswith('solve') and not line.startswith('output'):
                    new_lines.append(line)
                    new_lines.append('\n')

        new_lines.append('\n')
        new_lines.append('var int: obj = sum(V);\n')
        new_lines.append(f'solve :: int_search(X, {variable_heuristic}, {value_heuristic}) minimize obj;\n')
        new_lines.append('\n')

        new_lines.append('output ["Total violations: ", show(obj), "\\n"] ++\n')
        new_lines.append('["Solution: ", show(X), "\\n"] ++\n')
        new_lines.append('["Violations: ", show(V), "\\n"];\n')

        result = ''.join(new_lines)
        self.writeToFile("relaxed0001.mzn", result)
        return result

    def writeToFile(self, fileName, content):
        with open(fileName, "w") as f:
            f.write(content)

    def parse_and_display_constraints(self, file):
        constraints = []

        with open(file, 'r') as fileContent:
            for line in fileContent:
                if line.strip().startswith("constraint"):
                    constraints.append(line.strip()[11:].rstrip(";"))

        self.constraints_table.setRowCount(len(constraints))
        for i, constraint in enumerate(constraints):
            item = QTableWidgetItem(constraint)
            self.constraints_table.setItem(i, 0, item)

    def solver_finished(self):
        answer = self.solver_thread.answer
        solving_time = self.solver_thread.solving_time
        print(answer, solving_time) 
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.execution_timer.stop()
        self.state.setText("State: Sleeping")

    @staticmethod
    def solve(file_path, variable_heuristic, value_heuristic, instance, solver, timeout=None, output_callback=None, solver_thread=None):
        start = time.time()
        try:
            if solver == "minicp":
                command = Solvers["cp"][solver]["cmd"].format(instance=file_path, relaxation="BOOLEAN", variable=variable_heuristic, value=value_heuristic)
                print(command)
                p = subprocess.Popen(shlex.split(command),
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, universal_newlines=True)
            else:
                p = subprocess.Popen(shlex.split(Solvers["cp"][solver]["cmd"]),
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, universal_newlines=True)
            
            if solver_thread: 
                solver_thread.process = p
                
            p.stdin.write(instance)
            p.stdin.close()

            deadline = time.time() + timeout if timeout is not None else None

            accumulated_output = ""

            while True:
                if deadline is not None and time.time() > deadline:
                    parent = psutil.Process(p.pid)
                    for child in parent.children(recursive=True):
                        child.kill()
                    parent.kill()
                    break

                ready, _, _ = select.select([p.stdout], [], [], 1)

                if ready:
                    line = p.stdout.readline()
                    if not line:
                        break

                    accumulated_output += line

                    if line.strip() == "----------":
                        if output_callback is not None:
                            output_callback(accumulated_output)
                        accumulated_output = ""

            if deadline is not None and time.time() > deadline:
                remaining_output = p.stdout.read()
                if output_callback is not None:
                    output_callback(remaining_output)

                answer = "?"

            else:
                p.wait()
                if p.returncode != 0:
                    answer = "E"
                else:
                    answer = ""
        except subprocess.CalledProcessError as e:
            print(f"{e}")
            answer = "E"

        end = time.time()
        solving_time = end - start

        return answer, solving_time

    def write_to_console(self, text):
        cursor = self.console_output.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.console_output.setTextCursor(cursor)
        self.console_output.ensureCursorVisible()
        if "Violations:" in text:
            violations = self.parse_violations(text)
            self.update_constraints_table(violations)
            
            # Update the violation_data and time_data lists
            total_violations = sum(violations)
            elapsed_time = time.time() - self.start_time
            self.violation_data.append(total_violations)
            self.time_data.append(elapsed_time)
            
            # Update the plot
            self.plot_data.setData(self.time_data, self.violation_data, symbol='o', symbolSize=5, symbolBrush=('b'))


    def parse_violations(self, output):
        lines = output.strip().split('\n')
        for line in lines:
            if line.startswith('Violations:'):
                line = line.replace('[', '').replace(']', '')
                return [int(x) for x in line[12:].strip().split(', ')]
        return []

    def update_constraints_table(self, violations):
        self.constraints_table.setRowCount(len(violations))
        for i, violation in enumerate(violations):
            if violation == 0:
                color = QColor(200, 255, 200)
            else:
                color = QColor(255, 200, 200)

            self.constraints_table.item(i, 0).setBackground(color)
            total_violations = sum(violations) 
            self.total_violations_label.setText(f"Total violations: {total_violations}")

class EmittingStream(QIODevice):
    text_written = pyqtSignal(str)

    def __init__(self, text_written=None):
        super(EmittingStream, self).__init__()
        self.text_written.connect(text_written)

    def write(self, data: str) -> int:
        self.text_written.emit(data)
        return len(data)
    
class SolverThread(QThread):
    output_received = pyqtSignal(str)

    def __init__(self, instance, solver, file_path, variable, value, timeout=None):
        super(SolverThread, self).__init__()
        self.instance = instance
        self.solver = solver
        self.file_path = file_path
        self.variable = variable
        self.value = value
        self.timeout = timeout
        self.process = None
        self.start_time = None

    def run(self):
        self.start_time = time.time()
        print(self.solver)
        try:
            self.answer, self.solving_time = MiniZincGUI.solve(self.file_path, self.variable, self.value, self.instance, self.solver, self.timeout, output_callback=lambda line: self.output_received.emit(line), solver_thread=self)
        except Exception as e:
            print(f"Error occurred while solving: {e}")
            self.answer = "E"
            self.solving_time = 0

if __name__ == "__main__":
    app = QApplication(sys.argv)
    minizinc_gui = MiniZincGUI()
    sys.exit(app.exec_())


    
   
