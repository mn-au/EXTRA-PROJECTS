#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/* Game Constants */
#define NUM_STATES 21       /* Total states from 0 to 20 */
#define MAX_BUFFER_SIZE 20  /* Maximum size for input buffer */

/* Game State Variables */
int currentScore = 0;                   /* Current score in the game */
char inputBuffer[MAX_BUFFER_SIZE];      /* Input buffer for user input */

/* Reinforcement Learning Variables */
double stateReward[NUM_STATES] = {0};   /* Reward for reaching a particular state */
double stateValue[NUM_STATES] = {0};    /* Estimated value of each state */
int bestAction[NUM_STATES] = {0};       /* Best action to take at each state */
double discountFactor = 0.5;            /* Discount factor for future rewards */

/* Function Prototypes */
void initializeRewards(void);
void performValueIteration(void);
void determineBestActions(void);
int userMove(void);
int computerMove(void);
void printStateValues(void);
void printBestActions(void);
int main(void);

/* Main function */
int main(void)
{
    srand(time(NULL)); /* Seed the random number generator */
    int playerChoice;

    /* Initialize rewards and perform value iteration */
    initializeRewards();
    performValueIteration();
    determineBestActions();


    /* Uncomment the following lines to print computed values and actions */
    printStateValues();
    printBestActions();

    printf("Welcome to the WhoSays20 Game!\n\n");
    currentScore = 0;
    playerChoice = 0;

    /* Decide who goes first */
    while (playerChoice != 1 && playerChoice != 2) {
        printf("Who goes first? Enter 1 for you or 2 for computer: ");
        fgets(inputBuffer, sizeof(inputBuffer), stdin);
        playerChoice = atoi(inputBuffer);
    }

    /* If computer goes first */
    if (playerChoice == 2) {
        currentScore += computerMove();
    }

    /* Game loop */
    while (currentScore < 20) {
        currentScore += userMove();
        if (currentScore >= 20) {
            printf(" THE USER WINS!!!\n");
            break;
        }
        currentScore += computerMove();
        if (currentScore >= 20) {
            printf("COMPUTER WIN!!!\n");
            break;
        }
    }

    return 0;
}

/* Function Definitions */

/* User's move function */
int userMove(void)
{
    int userInput = 0;
    while (userInput != 1 && userInput != 2) {
        printf("We are at %d, add 1 or 2? ", currentScore);
        fgets(inputBuffer, sizeof(inputBuffer), stdin);
        userInput = atoi(inputBuffer);
    }
    return userInput;
}

/* Computer's move function using the best action determined by value iteration */
int computerMove(void)
{
    int move = bestAction[currentScore];
    printf("We are at %d, the computer adds %d.\n", currentScore, move);
    return move;
}

/* Initialize rewards for each state */
void initializeRewards(void)
{
    int state;
    for (state = 0; state <= 20; state++) {
        stateReward[state] = 0.0;
    }
    stateReward[20] = 10.0;  /* Reward for winning */
    stateReward[19] = -10.0; /* Penalty for losing */
    stateReward[18] = -5.0;  /* Risky state */
}

/* Perform value iteration to estimate the value of each state */
void performValueIteration(void)
{
    int state;
    /* Initialize values for terminal states */
    stateValue[20] = -10.0; /* Loss state */
    stateValue[19] = 10.0;  /* Winning move by adding 1 */
    stateValue[18] = 10.0;  /* Winning move by adding 2 */

    /* Perform value iteration for other states */
    for (state = 17; state >= 0; state--) {
        /* Check if the opponent has an advantage */
        if (stateValue[state + 1] > stateValue[state + 3] &&
            stateValue[state + 2] > stateValue[state + 3]) {
            /* Opponent has advantage regardless of your move */
            stateValue[state] = discountFactor * stateValue[state + 3];
        } else {
            /* You have the advantage */
            stateValue[state] = (stateValue[state + 1] > 0)
                                    ? stateValue[state + 1]
                                    : -stateValue[state + 1];
        }
    }
}

/* Determine the best action at each state based on the computed values */
void determineBestActions(void)
{
    int state;
    for (state = 0; state < 20; state++) {
        if (stateValue[state + 1] <= stateValue[state + 2]) {
            bestAction[state] = 1;
        } else {
            bestAction[state] = 2;
        }
    }
}

/* Function to print the values of each state */
void printStateValues(void)
{
    int state;
    printf("\nState Values:\n");
    printf("State\tValue\n");
    printf("-----------------\n");
    for (state = 0; state <= 20; state++) {
        printf("%d\t%.2f\n", state, stateValue[state]);
    }
}

/*  Function to print the best action at each state */
void printBestActions(void)
{
    int state;
    printf("\nBest Actions at Each State:\n");
    for (state = 0; state < 20; state++) {
        printf("At state %2d, best action is to add %d.\n", state, bestAction[state]);
    }
}
