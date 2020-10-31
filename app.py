import flask
from maze2d import Maze2D, Maze2DSolver, MazeSettings, MazeSolverSettings
from flask import jsonify, request

app = flask.Flask(__name__)
#app.config["DEBUG"] = True
VERSION = "1.0.0"

@app.route('/maze2d/solve', methods=['POST'])
def solve_maze2d():
    MAX_STEPS = 'max_steps'
    REWARDS_MATRIX = 'rewards_matrix'
    GAMMA = 'gamma'
    HISTORY = 'history'
    TERMINATION_THRESHOLD = 'termination_threshold'

    try:
        #Initialize Environment
        if(MAX_STEPS in request.json):
            max_steps:int = int(request.json[MAX_STEPS])
        else:
            raise Exception("Required field missing -> '%s'" % MAX_STEPS)

        if(REWARDS_MATRIX in request.json):
            reward_matrix: [[]] = request.json[REWARDS_MATRIX]
        else:
            raise Exception("Required field missing -> '%s'" % REWARDS_MATRIX)
            
        maze_settings = MazeSettings(reward_matrix,max_steps)
        maze = Maze2D(maze_settings)

        #Initialize Solver
        gamma: float = None
        threshold: float = None
        history: float = None
        if(GAMMA in request.json):
            gamma = request.json[GAMMA]
        if(HISTORY in request.json):
            history = request.json[HISTORY]
        if(TERMINATION_THRESHOLD in request.json):
            threshold = request.json[TERMINATION_THRESHOLD]
        
        solver_settings = MazeSolverSettings(gamma,threshold,history)
        solver = Maze2DSolver(maze,solver_settings)
        solution = solver.solve()
        solution["version"] = VERSION
        return jsonify(solution)
    except Exception as err:
        return jsonify({'error':str(err)})

if __name__ == '__main__':
    app.run()