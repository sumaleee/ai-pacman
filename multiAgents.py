# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        currentGameState: The current search state

        action: Direction; The action taken

        returns: float; a heuristic for the given (state,action) pair

        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodList = newFood.asList()
        score = successorGameState.getScore()

        # make pacman move towards closest food
        if foodList:
            foodDistances = [manhattanDistance(newPos, food) for food in foodList]
            score += 10.0 / min(foodDistances)
        
        # penalize remaining food
        score = score - 4 * len(foodList)

        # evaluate ghost positions
        for ghost in newGhostStates:
            ghostPos = ghost.getPosition()
            distance = manhattanDistance(newPos, ghostPos)
            if ghost.scaredTimer > 0:
                # if ghost is scared, move towards it to eat it
                score = score + 5.0 / (distance + 1)
            else:
                # if ghost is active, dangerous
                if distance < 2:
                    score = score - 500  # big penalty for being too close
                else:
                    score = score - 2.0 / distance

        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent & AlphaBetaPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 7)
    """

    def getAction(self, gameState):
        """
        gameState: the current state

        returns: Direction; the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        def minimax(state, agent, depth):
                # terminal condition is win, lose or max search depth reached
                if state.isWin() or state.isLose() or depth == self.depth:
                    return self.evaluationFunction(state)
                
                # pacman's move, maximize
                if agent == 0:
                    value = -float('inf')
                    for action in state.getLegalActions(agent):
                        successor = state.generateSuccessor(agent, action)
                        value = max(value, minimax(successor, 1, depth))
                    return value
                else:
                    # determine the next agent and whether we should increase the depth
                    nextAgent = agent + 1
                    nextDepth = depth
                    if nextAgent == state.getNumAgents():
                        nextAgent = 0
                        nextDepth = depth + 1
                    value = float('inf')
                    for action in state.getLegalActions(agent):
                        successor = state.generateSuccessor(agent, action)
                        value = min(value, minimax(successor, nextAgent, nextDepth))
                    return value

        bestAction = None
        bestValue = -float('inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = minimax(successor, 1, 0)
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 8)
    """

    def getAction(self, gameState):
        """
        gameState: the current state

        returns: Direction; the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(state, agent, depth):
            # terminal condition is win, lose, or max search depth reached
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            numAgents = state.getNumAgents()
            # once all agents have moved, start new ply
            if agent >= numAgents:
                return expectimax(state, 0, depth + 1)
            
            legalActions = state.getLegalActions(agent)
            if not legalActions:
                return self.evaluationFunction(state)
            
            if agent == 0:
                # pacman's move, maximize
                bestValue = -float('inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agent, action)
                    bestValue = max(bestValue, expectimax(successor, agent + 1, depth))
                return bestValue
            else:
                # ghosts' turn, compute expected value
                total = 0
                for action in legalActions:
                    successor = state.generateSuccessor(agent, action)
                    total = total + expectimax(successor, agent + 1, depth)
                return total / len(legalActions)

        bestAction = None
        bestScore = -float('inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = expectimax(successor, 1, 0)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction


def betterEvaluationFunction(currentGameState):
    """
    currentGameState: the current state

    returns: float; the evaluation of the state

    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 9).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
def betterEvaluationFunction(currentGameState):
    """
      Your extreme, unstoppable evaluation function (question 9).
    """
    from util import manhattanDistance
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    powerPellet = currentGameState.getCapsules()
    score = currentGameState.getScore()
    
    # reward for being near the closest food, penalize for remaining pellets
    foodScore = 0
    if foodList:
        minFoodDistance = min(manhattanDistance(pacmanPos, food) for food in foodList)
        foodScore = 10.0 / (minFoodDistance + 1) - 4 * len(foodList)
    
    # make pacman move closer to food by penalizing total distance to all food
    totalFoodScore = 0
    if foodList:
        totalDistance = sum(manhattanDistance(pacmanPos, food) for food in foodList)
        totalFoodScore = -0.1 * totalDistance
    
    # reward for eating power pellets
    ppScore = 0
    if powerPellet:
        minPelletDistance = min(manhattanDistance(pacmanPos, pp) for pp in powerPellet)
        ppScore = 10.0 / (minPelletDistance + 1) - 4 * len(powerPellet)
    
    # avoid active ghosts by penalizing for being too close, but approach ghosts if theyre scared
    ghostScore = 0
    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        distance = manhattanDistance(pacmanPos, ghostPos)
        if ghost.scaredTimer > 0:
            ghostScore += 2.0 / (distance + 1)
        else:
            if distance < 2:
                ghostScore -= 500 
            else:
                ghostScore -= 2.0 / distance
    
    # penalize moving into states with few legal moves, avoid getting trapped
    legalActions = currentGameState.getLegalActions(0)
    trapPenalty = 0
    if len(legalActions) < 3:
        trapPenalty = 100

    return score + foodScore + totalFoodScore + ppScore + ghostScore - trapPenalty
# Abbreviation
better = betterEvaluationFunction

