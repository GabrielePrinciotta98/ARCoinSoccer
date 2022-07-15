#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <opencv/cv.hpp>

#include "DrawPrimitives.h"
#include "MarkerTracker.h"
#include "CoinTracker.h"
#include "PoseEstimation.h"
#include "ObjLoader.h"
#include "ParticleSystem.h"

// Windows libraries for sound effects
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(_WIN64)
#include <windows.h>
#include <mmsystem.h>
#endif
#include <GL/freeglut.h>
#include <GL/glut.h>

#define FREEGLUT_STATIC
#define _LIB
#define FREEGLUT_LIB_PRAGMAS 0

//#define CALIBRATION_HINT // Show a quad on the marker to adjust fov
//#define CAMERA 1 // Use camera
#define VIDEO 3 // If camera is disabled, use specified video

using namespace cv;
using namespace std;

//callbacks
void reshape(GLFWwindow* window, int width, int height);
void onKey(GLFWwindow* window, int key, int scancode, int action, int mods);
//openGL initialization
void initGL(int argc, char* argv[]);
//rendering
void drawScene(float* markerMatrix, vector<Coin>& coins, bool ball, bool marker, float time);
void drawOverlays(float* markerMatrix, vector<Coin>& coins, bool ball, bool marker, float time);
void drawUI(float time);

void display(GLFWwindow* window, Mat &img_background, float* markerMatrix, vector<Coin>& coins, bool ball, bool marker);

// Tracker parameters for our test videos...

#ifdef CAMERA
double fov = 40;
double downscale = 0.5;
CoinTracker coinTracker(10, 10, 10, 40.0, 30.0);

#elif VIDEO == 0
std::string videoFile = "CoinSoccerMovie.mp4";
double fov = 63;
double downscale = 0.25;
CoinTracker coinTracker(6, 40.0, 30.0);

#elif VIDEO == 1
std::string videoFile = "CoinSoccerMovie2.mp4";
double fov = 80;
double downscale = 1.0 / 3;
CoinTracker coinTracker(12, 12, 15, 40.0, 30.0);

#elif VIDEO == 2
std::string videoFile = "CoinSoccerMovie3.mp4";
double fov = 59;
double downscale = 0.5;
CoinTracker coinTracker(6, 6, 11, 28.0, 28.0);

#elif VIDEO == 3
std::string videoFile = "CoinSoccerMovie4.mp4";
double fov = 81;
double downscale = 0.5;
CoinTracker coinTracker(9, 9, 15, 40.0, 30.0);

#elif VIDEO == 4
std::string videoFile = "CoinSoccerMovie5.mp4";
double fov = 59;
double downscale = 0.5;
CoinTracker coinTracker(9, 9, 15, 40.0, 30.0);
#endif

int cameraWidth, cameraHeight;
int windowWidth, windowHeight;
int savedWWidth, savedWHeight, savedWPosX, savedWPosY;
double prevFov = fov;

// OpenGL shadow-mapping stuff
const bool shadowMapping = true;
const GLsizei shadowMapSize = 1024;
GLuint shadowMapTexture, shadowFrameBuffer;
float lightProjectionMatrix[16], lightViewMatrix[16], textureMatrix[16];
float cameraProjectionMatrix[16], cameraViewMatrix[16];

float smoothedZ = 100; // z-value at which 3d-overdraws are drawn (smoothed over time)
int goals = 0; // score

// 3d-models and textures
GLuint vignetteTexture;
ObjModel soccerModel("soccer_ball.obj", "soccer_ball_diffuse.png"), 
shoeModel("football_boots.obj", { Vec4f(1, 1, 1, 1), Vec4f(0.2, 0.2, 0.2, 1), Vec4f(1, 1, 1, 1), Vec4f(0.8, 0.8, 0.8, 1), Vec4f(0.2, 0.2, 0.2, 1), Vec4f(0, 0, 0, 1) }),
coinModel("coin.obj", { Vec4f(1.0, 0.75, 0.3, 1.0), Vec4f(0.8, 0.8, 0.8, 1.0) });
Texture titleTex("title.png"), creditTex("credits.png"), anyKeyTex("anykey.png"), moveCoinTex("moveCoins.png"), noMarkerTex("noMarker.png"), noTrackingTex("noTracking.png"), winTex("victory.png"), circleTex("circle.png"), goalTex("goal.png"), stripeTex("stripe.png", true);
ParticleSystem uiSystem(""), ballFxSystem("Cloud.png"), goalFxSystem("");

// Starting positions of coins on title-screen
const bool startPos = true;
Vec3f startPositions[] = { Vec3f(0, -0.5f, 0), Vec3f(-0.15f, -0.25f, 0), Vec3f(0.3f, -0.3f, 0) };

const Vec2f targetAreaMin(-0.5, 0.65), targetAreaMax(0.5, 0.9);
const int goalsUntilFinalArea = 3;

bool fullscreen = false;
int framesWithoutMarker = 100;
int framesWithoutCoins  = 100;

enum class GameState
{
    TITLE,
    GAME,
    VICTORY
};

GameState gameState = GameState::TITLE;
float lastGameStateChange; // For animation

bool ballMoving;
Vec3f coinMoveStartPos;
float lastGoal = -100; // For animation

float sqDist(Vec2f pos1, Vec2f pos2)
{
    float dx = pos1[0] - pos2[0];
    float dy = pos1[1] - pos2[1];
    return dx * dx + dy * dy;
}

// Used for tracking
int findNewCoinClosestTo(std::vector<Coin> &coins, Vec2f lastPosition, float maxDist)
{
    float closestDist = 0;
    int closest = -1;
    for (int i = 0; i < coins.size(); ++i)
    {
        if (coins[i].framesTracked == 0) // coin found in this frame
        {
            // Close to last position (where we lost tracking)
            float dist = sqDist(coins[i].pos2D, lastPosition);
            if ((closest == -1 || dist < closestDist) && dist < maxDist)
            {
                closestDist = dist;
                closest = i;
            }
        }
    }
    return closest;
}

// Used for detecting goals
bool lineSegmentIntersection(Vec2f p0, Vec2f p1, Vec2f p2, Vec2f p3)
{
    Vec2f s1 = p1 - p0;
    Vec2f s2 = p3 - p2;
    float s = (-s1[1] * (p0[0] - p2[0]) + s1[0] * (p0[1] - p2[1])) / (-s2[0] * s1[1] + s1[0] * s2[1]);
    float t = (s2[0] * (p0[1] - p2[1]) - s2[1] * (p0[0] - p2[0])) / (-s2[0] * s1[1] + s1[0] * s2[1]);

    return s >= 0 && s <= 1 && t >= 0 && t <= 1;
}

int main(int argc, char* argv[])
{
    //Set up OpenCV
#ifdef CAMERA
    cv::VideoCapture cap(1);
#else
    cv::VideoCapture cap(videoFile);
#endif

    if (!cap.isOpened())
    {
        std::cout << "No capture" << std::endl;
        return -1;
    }

    // Setup GLFW
    GLFWwindow* window;

    if (!glfwInit())
        return -1;

    // Setup GLUT
    glutInit(&argc, argv);

    glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);  // Map between sRGB and Linear-space for better lighting
    glfwWindowHint(GLFW_SAMPLES, 8); // Anti-aliasing
   //glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    cameraWidth = cap.get(CAP_PROP_FRAME_WIDTH);
    cameraHeight = cap.get(CAP_PROP_FRAME_HEIGHT);
    window = glfwCreateWindow(cameraWidth, cameraHeight, "Coin Soccer", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    glfwSetFramebufferSizeCallback(window, reshape);
    glfwSetKeyCallback(window, onKey);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // Setup GLEW
    if (glewInit() != GLEW_OK)
    {
        glfwTerminate();
        return -1;
    }

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    reshape(window, width, height);
    initGL(argc, argv); // Main init

    const double kMarkerSize = 0.045;
    MarkerTracker markerTracker(kMarkerSize, 120, 120);
    coinTracker.init();

    // Coin detection
    std::vector<Coin> lastCoins;
    std::vector<Coin> coins;
    // Game logic
    cv::Vec2f lastBallPosition(0, 0);
    // Only valid while coin is moving
    cv::Vec2f flag1Position(0, 0);
    cv::Vec2f flag2Position(0, 0);

    int framesWithoutMovement = 60;
    int framesWithoutFlags = 0;
    int framesBallMoving = 0;
    int framesWithoutBall = 0;
    bool detectedGoalThisTurn = false;
    size_t ballID = -1, flag1Id = -1, flag2Id = -1;

    cv::Mat img_background, img_background_resized, img_background2;
    float resultMatrix[16];
    while (!glfwWindowShouldClose(window))
    {
        // Live update for slider
        if (fov != prevFov)
            reshape(window, windowWidth, windowHeight);

        if (!cap.read(img_background))
        {
            cap.set(CAP_PROP_POS_FRAMES, 0); // Loop back to frame 0
            if (!cap.read(img_background))
                break;
        }

        img_background2 = img_background.clone();
        lastCoins = coins;

        // Downscale the image before looking for coins for better quality/performance 
        cv::resize(img_background, img_background_resized, Size(), downscale, downscale, CV_INTER_LINEAR);
        coinTracker.findCoins(img_background_resized, lastCoins, coins);
        bool marker = markerTracker.findMarker(img_background, resultMatrix);
        
        if (marker)
            framesWithoutMarker = 0;
        else
            framesWithoutMarker++;

        // xyz Position from last markerMatrix
        double x = resultMatrix[3];
        double y = resultMatrix[7];
        double z = resultMatrix[11] - 0.01;
        smoothedZ = abs(smoothedZ - z) > 1 ? z : smoothedZ * 0.9 + z * 0.1;
        // For mapping 2d pixel coordinates to 3d coordinates, adjust for (downscaled) resolution and perspective
        double scale = 2.0 * tan(fov / 180.0 * M_PI * 0.5) * (-smoothedZ) / ((double)img_background.rows * downscale);

        // We're only looking for 3 coins, 2 flags + ball
        if (!ballMoving && coins.size() > 3)
            coins.resize(3);

        // Map 2d-pixel coordinates for each coin to 3d
        for (int i = 0; i < coins.size(); ++i)
        {
            Coin& c = coins[i];
            c.pos3D[0] = (c.pos2D[0] - img_background.cols * downscale * 0.5) * scale;
            c.pos3D[1] = (img_background.rows * downscale * 0.5 - c.pos2D[1]) * scale;
            c.pos3D[2] = smoothedZ;
        }

        int ball = -1;

        // Game logic for detecting goals
        if (gameState == GameState::GAME)
        {
            // 
            bool prevBallMoving = ballMoving;

            float closestDist = 0;
            int closest = -1;
            float fastestVel = 0;
            int fastest = -1;

            int flag1 = -1, flag2 = -1;

            // Find the fastest moving coin, and the coin closest to the marker position
            for (int i = 0; i < coins.size(); ++i)
            {
                Coin& c = coins[i];

                float vel = c.vel[2];
                float dist = sqDist(lastBallPosition, c.pos2D);
                // For a bit better tracking of the ball, exclude far away coins and flags
                bool plausibleCandidate = (!ballMoving || dist < coinTracker.coinSize * coinTracker.coinSize * 12 * 12 && c.id != flag1Id && c.id != flag2Id);
				if ((fastest == -1 || fastestVel < vel) && plausibleCandidate)
                {
                    fastestVel = vel;
                    fastest = i;
                }

                float dx = c.pos3D[0] - x;
                float dy = c.pos3D[1] - y;
                dist = dx * dx + dy * dy;
                if ((closest == -1 || dist < closestDist) && c.framesTracked > 3 && i < 3)
                {
                    closestDist = dist;
                    closest = i;
                }

                // Track ball and flags by id (only while ball is moving)
                if (c.id == ballID)
                    ball = i;
                else if (c.id == flag1Id)
                    flag1 = i;
                else if (c.id == flag2Id)
                    flag2 = i;
            }

            // We got enough coins for 2 flags and a ball
            if (coins.size() >= 3)
            {
                if (ballMoving)
                {
                    framesBallMoving++;

                    // If we lost track of the flags, try to find some coin close to their original position
                    if (flag1 == -1)
                    {
                        flag1Id = -1;
                        flag1 = findNewCoinClosestTo(coins, flag1Position, coinTracker.coinSize * coinTracker.coinSize);
                    }
                    if (flag2 == -1)
                    {
                        flag2Id = -1;
                        flag2 = findNewCoinClosestTo(coins, flag2Position, coinTracker.coinSize * coinTracker.coinSize);
                    }

                    if (flag1 == -1 || flag2 == -1)
                    {
                        // Tracking lost
                        if (framesWithoutFlags++ > 5)
                            ballMoving = false;
                    }
                    else
                    {
                        flag1Id = coins[flag1].id;
                        flag2Id = coins[flag2].id;
                        framesWithoutFlags = 0;

                        if (ball == -1)
                        {
                            // We have lost track of the ball, 
                            // look for fastest coin,
                            // coin reasonably close to the last tracked position or 
                            // in the worst case just some coin that has been tracked stably for a few frames
                            if (fastestVel > 20)
                                ball = fastest;
                            else
                                ball = findNewCoinClosestTo(coins, lastBallPosition, coinTracker.coinSize * coinTracker.coinSize * 12 * 12);
                            
                            if (ball == -1 && framesWithoutBall > 5 && coins[2].framesTracked > 3)
                                ball = 2;

                            if (ball != -1)
                            {
                                framesWithoutBall = 0;
                                framesWithoutMovement = 0;
                            }
                            else
                                framesWithoutBall++;
                        }

                        // If the ball is very close to one of the flag positions, this is probably a mistake
                        if (ball != -1 && (sqDist(coins[ball].pos2D, flag1Position) < coinTracker.coinSize * coinTracker.coinSize * 0.25
                            || sqDist(coins[ball].pos2D, flag2Position) < coinTracker.coinSize * coinTracker.coinSize * 0.25))
                        {
                            ball = -1;
                            ballID = (size_t) -1;
                            std::cout << "Ball at flag position\n";
                        }

                        if (fastestVel <= 20 && ball != -1)
                            framesWithoutMovement++;
                    }

                    // Nothing seems to be moving anymore
                    if (framesWithoutMovement > 5)
                        ballMoving = false;
                    // Timeout
                    if (framesWithoutBall > 15)
                        ballMoving = false;
                }
                else
                {
                    // One of the coins has started moving
                    if (fastestVel > 20)
                    {
                        ball = fastest;
                        lastBallPosition = Vec2f(coins[ball].pos2D[0] - coins[ball].vel[0], coins[ball].pos2D[1] - coins[ball].vel[1]);
                        ballMoving = true;
                    }
                    // We have lost track of the ball... maybe it started moving?
                    else if (ball == -1)
                    {
                        ball = findNewCoinClosestTo(coins, lastBallPosition, coinTracker.coinSize * coinTracker.coinSize);
                        if (ball != -1 || coins[1].framesTracked > 3 && coins[2].framesTracked == 0)
                            ballMoving = true;
                    }
                }

                // Nothing is moving, ball is the coin closest to the marker 
                if (!ballMoving)
                    ball = closest;

                // Ensure the coin with the ball is always at coins[2], flags at index 0 and 1 (for convenience)
                if (flag1 != -1 && flag1 != 0)
                {
                    swap(coins[0], coins[flag1]);
                    if (ball == 0)
                        ball = flag1;
                    if (flag2 == 0)
                        flag2 = flag1;
                    flag1 = 0;
                }
                if (flag2 != -1 && flag2 != 1)
                {
                    swap(coins[1], coins[flag2]);
                    if (ball == 1)
                        ball = flag2;
                    flag2 = 1;
                }
                if (ball != -1 && ball != 2)
                {
                    swap(coins[2], coins[ball]);
                    ball = 2;
                }
            }
            // We have less than 3 coins detected
            else 
            {
                // Assume the player has hit the ball, if the two other coins are reasonably stationary
                if (coins.size() == 2 && coins[1].framesTracked > 3 && coins[0].vel[2] < 20 && coins[1].vel[2] < 20)
                    ballMoving = true;

                ball = -1;
                ballID = -1;
                framesWithoutMovement = 0;
            }

            // Keep track of the ball's position when it first moved (the player hits it)
            if (ballMoving)
            {
                framesBallMoving++;
                if (!prevBallMoving)
                {
                    coinMoveStartPos[0] = (lastBallPosition[0] - img_background.cols * downscale * 0.5) * scale;
                    coinMoveStartPos[1] = (img_background.rows * downscale * 0.5 - lastBallPosition[1]) * scale;
                    coinMoveStartPos[2] = smoothedZ;

                    flag1Id = coins[0].id;
                    flag2Id = coins[1].id;
                    flag1Position = coins[0].pos2D;
                    flag2Position = coins[1].pos2D;
                    detectedGoalThisTurn = false;
                    framesBallMoving = 0;
                    framesWithoutMovement = 0;
                    framesWithoutBall = 0;

                    //std::cout << "Ball started moving\n";
                }
            }
            else if (prevBallMoving)
            {
                flag1Id = flag2Id = -1;
                framesBallMoving = 0;
                //std::cout << "Ball stopped moving\n";
            }

            // We have found the ball
            if (ball != -1)
            {
                ballID = coins[ball].id;
                const Vec2f& ballPos = coins[ball].pos2D;

                // Try to detect goals
                if (ballMoving && !detectedGoalThisTurn)
                {
                    const Vec2f& pos1 = flag1Position; // Flag position 1
                    const Vec2f& pos2 = flag2Position; // Flag position 2

                    // Signed distance of ball to the line between the two flags
                    float lastDist = (pos2[1] - lastBallPosition[1]) * (pos2[0] - pos1[0]) - (pos2[0] - lastBallPosition[0]) * (pos2[1] - pos1[1]);
                    float dist = (pos2[1] - ballPos[1]) * (pos2[0] - pos1[0]) - (pos2[0] - ballPos[0]) * (pos2[1] - pos1[1]);
                   
                    // Ball is on the opposite side of the flags compared to last time
                    if (signbit(lastDist) != signbit(dist))
                    {
                        bool betweenFlags = lineSegmentIntersection(lastBallPosition, ballPos, pos1, pos2); // The ball passes between the flags, not on to the sides

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(_WIN64)
                        PlaySound(TEXT(betweenFlags ? "goal.wav" : "goalMissed.wav"), NULL, SND_FILENAME | SND_ASYNC);
#endif
                        detectedGoalThisTurn = true;
                        if (betweenFlags)
                        {
                            cout << "Goal" << "\n";

                            goals++;
                            lastGoal = glfwGetTime();
                            goalFxSystem.emit(coins[0].pos3D, coins[1].pos3D, 20, 0);
                        }
                        else
                        {
                            cout << "Goal missed..." << "\n";
                            goals--;
                        }

                        // Draw red or green line between flags
                        Point start(cvRound(pos1[0] / downscale), cvRound(pos1[1] / downscale));
                        Point end(cvRound(pos2[0] / downscale), cvRound(pos2[1] / downscale));
                        line(img_background2, start, end, betweenFlags ? Scalar(0, 255, 0) : Scalar(0, 0, 255), 5);
                    }

                    /* Blue debug line
                    Point start(cvRound(ballPos[0] / downscale), cvRound(ballPos[1] / downscale));
                    Point end(cvRound(lastBallPosition[0] / downscale), cvRound(lastBallPosition[1] / downscale));
                    line(img_background2, start, end, Scalar(255, 0, 0), 5);
                    */
                }

                lastBallPosition = ballPos;
            }
            else
                ballID = -1;

            // test for final goal area
            if (goals > goalsUntilFinalArea && ball != -1)
            {
                float x = (coins[ball].pos2D[0] - img_background.cols * downscale * 0.5) / ((float)img_background.rows * downscale) * 2;
                float y = (img_background.rows * downscale * 0.5 - coins[ball].pos2D[1]) / ((float)img_background.rows * downscale) * 2;

                // ball has reached goal area
                if (x > targetAreaMin[0] && x < targetAreaMax[0] && y > targetAreaMin[1] && y < targetAreaMax[1])
                {
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(_WIN64)
                    PlaySound(TEXT("goal.wav"), NULL, SND_FILENAME | SND_ASYNC);
#endif
                    cout << "Victory!" << "\n";

                    lastGameStateChange = glfwGetTime();
                    gameState = GameState::VICTORY;
                }
            }
        }
        else // Not in game
        {
            // On the titlescreen detect if coins have been moved in the start-positions
            if (startPos && coins.size() >= 3 && gameState == GameState::TITLE)
            {
                for (int i = 0; i < 3; ++i)
                {
                    startPositions[i][2] = 0;

                    for (int j = 0; j < coins.size(); ++j)
                    {
                        Coin& c = coins[j];
                        c.pos3D[0] = (c.pos2D[0] - img_background.cols * downscale * 0.5) * scale;
                        c.pos3D[1] = (img_background.rows * downscale * 0.5 - c.pos2D[1]) * scale;
                        float dx = c.pos2D[0] - (startPositions[i][0] *  0.5 * img_background.rows * downscale + img_background.cols * downscale * 0.5);
                        float dy = c.pos2D[1] - (startPositions[i][1] * -0.5 * img_background.rows * downscale + img_background.rows * downscale * 0.5);
                        float dist = dx * dx + dy * dy;
                        if (dist < coinTracker.coinSize * coinTracker.coinSize && c.framesTracked > 2)
                        {
                            // I'm abusing the z-Position of the vector as a boolean
                            startPositions[i][2] = 1;
                            break;
                        }
                    }
                }

                // All coins at the start position, start-game
                if (startPositions[0][2] > 0.5 && startPositions[1][2] > 0.5 && startPositions[2][2] > 0.5)
                {
                    gameState = GameState::GAME;
                    lastGameStateChange = glfwGetTime();
                }
            }

            // Also draw a ball on the title-screen
            if (coins.size() >= 3)
                ball = 2;
        }

        // For showing a tracking-lost message
        if (coins.size() < 3 || coins[0].framesTracked < 3)
            framesWithoutCoins++;
        else
            framesWithoutCoins = 0;

        display(window, img_background2, resultMatrix, coins, ball != -1, marker);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    coinTracker.cleanup();
    glfwTerminate();
    return 0;
}

// Setup all-kinds of graphics stuff
void initGL(int argc, char* argv[])
{

    // ==== Lighting
    GLfloat zero[] = { 0, 0, 0, 1.0 };
    GLfloat one[] = { 1.0, 1.0, 1.0, 1.0 };
    GLfloat light_pos[] = { 0.5, 0.3, 1.0, 0 };
    GLfloat light_pos2[] = { -0.5, -0.3, -1.0, 0 };
    GLfloat light_pos3[] = { -0.5, 0.5, -0.1, 0 };
    GLfloat light_amb[] = { 0.075, 0.075, 0.075, 1.0 };
    GLfloat light_dif[] = { 0.9, 0.8, 0.75, 1.0 };
    GLfloat light_dif2[] = { 0.15, 0.15, 0.2, 1.0 };
    GLfloat light_spe[] = { 0.8, 0.8, 0.8, 1.0 };

    glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
    glLightfv(GL_LIGHT0, GL_AMBIENT, zero);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_dif);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_spe);

    glLightfv(GL_LIGHT1, GL_POSITION, light_pos2);
    glLightfv(GL_LIGHT1, GL_AMBIENT, light_amb);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, light_dif2);
    glLightfv(GL_LIGHT1, GL_SPECULAR, zero);

    glLightfv(GL_LIGHT2, GL_POSITION, light_pos3);
    glLightfv(GL_LIGHT2, GL_AMBIENT, zero);
    glLightfv(GL_LIGHT2, GL_DIFFUSE, light_dif);
    glLightfv(GL_LIGHT2, GL_SPECULAR, light_spe);

    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, light_spe);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 120.0);
    glEnable(GL_LIGHT1);

    glLightModelf(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE);
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, zero);
    glEnable(GL_COLOR_MATERIAL);

    // ==== For better graphics

    glEnable(GL_NORMALIZE);
    glEnable(GL_DITHER);
    glEnable(GL_FRAMEBUFFER_SRGB);

    // ==== Culling and depth

    glEnable(GL_CULL_FACE);
    glClearColor(0., 0., 0., 1);
    glClearDepth(1);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_DEPTH_TEST);

    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // === Shadowmapping with fixed-pipeline based on http://www.paulsprojects.net/tutorials/smt/smt.html
    glGenFramebuffers(1, &shadowFrameBuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, shadowFrameBuffer);

    glGenTextures(1, &shadowMapTexture);
    glBindTexture(GL_TEXTURE_2D, shadowMapTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, shadowMapSize, shadowMapSize, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, one);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE_ARB, GL_COMPARE_R_TO_TEXTURE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC_ARB, GL_LEQUAL);
    glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE_ARB, GL_INTENSITY);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadowMapTexture, 0);
    glDrawBuffer(GL_NONE);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "Framebuffer not complete\n";

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glOrtho(-0.3, 0.3, -0.3, 0.3, 1, 1.5);
    glGetFloatv(GL_MODELVIEW_MATRIX, lightProjectionMatrix);

    glLoadIdentity();
    gluLookAt(light_pos[0], light_pos[1], light_pos[2] - 0.25,
        0.0f, 0.0f, -0.25f,
        0.0f, 0.0f, 1.0f);
    glGetFloatv(GL_MODELVIEW_MATRIX, lightViewMatrix);

    static float biasMatrix[16] = {
        0.5f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.5f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.5f, 0.0f,
        0.5f, 0.5f, 0.5f, 1.0f };
    float tmpMatrix[16];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
        {
            float dot = 0;
            for (int k = 0; k < 4; ++k)
                dot += biasMatrix[k * 4 + j] * lightProjectionMatrix[i * 4 + k];
            tmpMatrix[i * 4 + j] = dot;
        }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
        {
            float dot = 0;
            for (int k = 0; k < 4; ++k)
                dot += tmpMatrix[k * 4 + j] * lightViewMatrix[i * 4 + k];
            textureMatrix[i * 4 + j] = dot;
        }
    //textureMatrix = biasMatrix * lightProjectionMatrix * lightViewMatrix;

    // vignette texture
    glGenTextures(1, &vignetteTexture);
    glBindTexture(GL_TEXTURE_2D, vignetteTexture);
    unsigned char* data = new unsigned char[512 * 512 * 4];
    for (int i = 0; i < 512; ++i)
        for (int j = 0; j < 512; ++j)
        {
            data[(i * 512 + j) * 4 + 0] = 0;
            data[(i * 512 + j) * 4 + 1] = 0;
            data[(i * 512 + j) * 4 + 2] = 0;
            float x = i / 256.0F - 1;
            float y = j / 256.0F - 1;
            data[(i * 512 + j) * 4 + 3] = (unsigned char)min(255.0F, 255 * pow(x * x + y * y, 0.8F) * 0.5F);
        }
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 512, 512, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    delete[] data;
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

    // ==== 3d-models and textures

    soccerModel.init();
    shoeModel.init();
    coinModel.init();

    titleTex.init(); 
    creditTex.init();
    anyKeyTex.init();
    moveCoinTex.init();
    winTex.init();
    circleTex.init();
    noMarkerTex.init();
    noTrackingTex.init();
    goalTex.init();
    stripeTex.init();

    // ==== Particlesystems

    float time = glfwGetTime();
    uiSystem.pos = cv::Vec3f(0, 0.25, 0);
    uiSystem.emitPerSecond = 15;
    uiSystem.minEmitRad = 0.4;
    uiSystem.maxEmitRad = 0.6;
    uiSystem.minRadVel = 0.5;
    uiSystem.maxRadVel = 0.75;
    uiSystem.fadeTime = 1;
    uiSystem.minLifeTime = 1;
    uiSystem.maxLifeTime = 2;
    uiSystem.minStartSize = uiSystem.minEndSize = 0.025;
    uiSystem.maxStartSize = uiSystem.maxEndSize = 0.03;
    uiSystem.startColor1 = uiSystem.endColor1 = Vec4f(1, 1, 1, 0.2F);
    uiSystem.startColor2 = uiSystem.endColor2 = Vec4f(1, 1, 1, 0.4F);
    uiSystem.init(time);

    ballFxSystem.emitPerSecond = 0;
    ballFxSystem.emitPerDistance = 100;
    ballFxSystem.gravity = Vec3f(0, 0, 0.05F);
    ballFxSystem.minRadVel = 0.005;
    ballFxSystem.maxRadVel = 0.01;
    ballFxSystem.fadeTime = 0.25;
    ballFxSystem.minLifeTime = 0.5f;
    ballFxSystem.maxLifeTime = 1.5f;
    ballFxSystem.minStartRot = 0;
    ballFxSystem.maxStartRot = 360;
    ballFxSystem.minRotVelocity = -180;
    ballFxSystem.maxRotVelocity = 180;
    ballFxSystem.minVel = Vec3f(0, 0, 0.01F);
    ballFxSystem.maxVel = Vec3f(0, 0, 0.015F);
    ballFxSystem.minStartSize = ballFxSystem.minEndSize = 0.005;
    ballFxSystem.maxStartSize = ballFxSystem.maxEndSize = 0.03;
    ballFxSystem.startColor1 = ballFxSystem.endColor1 = Vec4f(0.5, 0.5, 0.5, 0.2F);
    ballFxSystem.startColor2 = ballFxSystem.endColor2 = Vec4f(1, 1, 1, 0.4F);
    ballFxSystem.init(time);

    goalFxSystem.emitPerSecond = 0;
    goalFxSystem.minEmitRad = goalFxSystem.maxEmitRad = 0;
    goalFxSystem.minRadVel = 0.025;
    goalFxSystem.maxRadVel = 0.05;
    goalFxSystem.fadeTime = 0.25;
    goalFxSystem.minLifeTime = 1;
    goalFxSystem.maxLifeTime = 2;
    goalFxSystem.minRotVelocity = -180;
    goalFxSystem.maxRotVelocity = 180;
    goalFxSystem.minVel = Vec3f(0, 0, 0.01F);
    goalFxSystem.maxVel = Vec3f(0, 0, 0.015F);
    goalFxSystem.minStartSize = 0.001;
    goalFxSystem.maxStartSize = 0.004;
    goalFxSystem.minEndSize = goalFxSystem.maxEndSize = 0;
    goalFxSystem.startColor1 = goalFxSystem.endColor1 = Vec4f(0, 1, 0, 0.8F);
    goalFxSystem.startColor2 = goalFxSystem.endColor2 = Vec4f(1, 1, 0, 0.4F);
    goalFxSystem.init(time);

    int error = glGetError(); // error-checking
    if (error)
        std::cout << gluErrorString(error) << " in init\n";
}

// Main-rendering
void display(GLFWwindow* window, Mat &img_background, float* markerMatrix, vector<Coin>& coins, bool ball, bool marker)
{
    float time = glfwGetTime();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_FRAMEBUFFER_SRGB);

    // ========= Depth-pass (for shadow mapping)
    if (shadowMapping)
    {
        glMatrixMode(GL_PROJECTION);
        glLoadMatrixf(lightProjectionMatrix);

        glMatrixMode(GL_MODELVIEW);
        glLoadMatrixf(lightViewMatrix);
        glViewport(0, 0, shadowMapSize, shadowMapSize);

        glBindFramebuffer(GL_FRAMEBUFFER, shadowFrameBuffer);
        glClear(GL_DEPTH_BUFFER_BIT);

        glCullFace(GL_FRONT);
        glShadeModel(GL_FLAT);
        glColorMask(0, 0, 0, 0);
        drawScene(markerMatrix, coins, ball, marker, time);

        // If we don't use a custom framebuffer
        //glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, shadowMapSize, shadowMapSize);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glCullFace(GL_BACK);
        glShadeModel(GL_SMOOTH);
        glColorMask(1, 1, 1, 1);
    }

    // ========= Camera-image
    glViewport(0, 0, windowWidth, windowHeight);
    glDisable(GL_DEPTH_TEST);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, 1, 1, 0);

    float backgroundOffset = 0.5F - 0.5F * cameraWidth * windowHeight / cameraHeight / windowWidth;  // Center camera image in window
    if (backgroundOffset < 0)
    {
        glRasterPos2f(0, 0);
        glBitmap(0, 0, 0, 0, backgroundOffset * windowWidth, 0, NULL); // Really weird hack needed when offset < 0
    }
    else
        glRasterPos2f(backgroundOffset, 0);

    // drawing of the opencv image
    glDrawPixels(img_background.cols, img_background.rows, GL_BGR, GL_UNSIGNED_BYTE, img_background.data);

    // ========= Main-pass
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_FRAMEBUFFER_SRGB);

    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(cameraProjectionMatrix);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glLoadMatrixf(cameraViewMatrix);

    if (shadowMapping)
    {
        glActiveTexture(GL_TEXTURE1);
        float tmp[4];
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
                tmp[j] = textureMatrix[j * 4 + i];
            glTexGeni(GL_S + i, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
            glTexGenfv(GL_S + i, GL_EYE_PLANE, tmp);
            glEnable(GL_TEXTURE_GEN_S + i);
        }

        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, shadowMapTexture);
        glActiveTexture(GL_TEXTURE0);

        // ========= Shadows on ground

        glEnable(GL_BLEND);
        glBlendFunc(GL_ZERO, GL_CONSTANT_ALPHA);
        glBlendColor(1, 1, 1, 0.2);
        glEnable(GL_ALPHA_TEST);
        glAlphaFunc(GL_LESS, 0.99f);

        glBegin(GL_QUADS);
        glVertex3f(-1, -1, smoothedZ);
        glVertex3f(1, -1, smoothedZ);
        glVertex3f(1, 1, smoothedZ);
        glVertex3f(-1, 1, smoothedZ);
        glEnd();

        glDisable(GL_BLEND);
        glActiveTexture(GL_TEXTURE1);
        glDisable(GL_TEXTURE_2D);
        glActiveTexture(GL_TEXTURE0);
        glDisable(GL_ALPHA_TEST);

        // ========= Invisible plane for occlusion

        glColorMask(0, 0, 0, 0);
        glBegin(GL_QUADS);
        glVertex3f(-1, -1, smoothedZ);
        glVertex3f(1, -1, smoothedZ);
        glVertex3f(1, 1, smoothedZ);
        glVertex3f(-1, 1, smoothedZ);
        glEnd();

        if (marker)
        {
#ifdef CALIBRATION_HINT
            glColorMask(1, 1, 1, 1);
#endif
            glColor4f(1, 0.5F, 0.2F,1);
            glPushMatrix();
            glMultTransposeMatrixf(markerMatrix);
            glScalef(0.035F, 0.035F, 1.0F);
            glBegin(GL_QUADS);
            glVertex3f(-1, 1, 0);
            glVertex3f(1, 1, 0);
            glVertex3f(1, -1, 0);
            glVertex3f(-1, -1, 0);
            glEnd();
            glPopMatrix();
        }
        glColorMask(1, 1, 1, 1);

        // ========= Scene in shadow
        glEnable(GL_LIGHTING);
        drawScene(markerMatrix, coins, ball, marker, time);

        // =========  Scene in light
        glActiveTexture(GL_TEXTURE1);
        glEnable(GL_TEXTURE_2D);
        glActiveTexture(GL_TEXTURE0);
        glEnable(GL_LIGHT0);
        glEnable(GL_ALPHA_TEST);
        glAlphaFunc(GL_GEQUAL, 0.99f);

        drawScene(markerMatrix, coins, ball, marker, time);

        glDisable(GL_LIGHT0);
        glDisable(GL_LIGHTING);
        glDisable(GL_ALPHA_TEST);
        glActiveTexture(GL_TEXTURE1);
        glDisable(GL_TEXTURE_2D);
        glDisable(GL_TEXTURE_GEN_S);
        glDisable(GL_TEXTURE_GEN_T);
        glDisable(GL_TEXTURE_GEN_R);
        glDisable(GL_TEXTURE_GEN_Q);
        glActiveTexture(GL_TEXTURE0);
    }
    else // no shadow-mapping
    {
        // ========= Invisible plane for occlusion

        glColorMask(0, 0, 0, 0);
        glBegin(GL_QUADS);
        glVertex3f(-1, -1, smoothedZ);
        glVertex3f(1, -1, smoothedZ);
        glVertex3f(1, 1, smoothedZ);
        glVertex3f(-1, 1, smoothedZ);
        glEnd();

        if (marker)
        {
#ifdef CALIBRATION_HINT
            glColorMask(1, 1, 1, 1);
#endif
            glColor4f(1, 0.5F, 0.2F, 1);
            glPushMatrix();
            glMultTransposeMatrixf(markerMatrix);
            glScalef(0.035F, 0.035F, 1.0F);
            glBegin(GL_QUADS);
            glVertex3f(-1, 1, 0);
            glVertex3f(1, 1, 0);
            glVertex3f(1, -1, 0);
            glVertex3f(-1, -1, 0);
            glEnd();
            glPopMatrix();
        }

        glColorMask(1, 1, 1, 1);

        // ========= Default scene
        glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);

        drawScene(markerMatrix, coins, ball, marker, time);

        glDisable(GL_LIGHT0);
        glDisable(GL_LIGHTING);
    }

    // ========= Transparent overlays

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    drawOverlays(markerMatrix, coins, ball, marker, time);

    // ========= 2D overdraw

    glClear(GL_DEPTH_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    float ratio = windowWidth / (float)windowHeight;
    glOrtho(-ratio, ratio, -1.0f, 1.0f, -10, 10);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    drawUI(glfwGetTime());

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);

    int error = glGetError();
    if (error)
        std::cout << gluErrorString(error) << " in display\n";
}

void drawScene(float* markerMatrix, vector<Coin>& coins, bool ball, bool marker, float time)
{
    // Draw shoe
    if (marker)
    {
        glPushMatrix();
        glMultTransposeMatrixf(markerMatrix);
        glColor4f(1.0, 1.0, 1.0, 1.0);

        glScalef(0.075F, 0.075F, 0.075F);
        glRotatef(-180, 1, 0, 0);
#if !CAMERA && VIDEO == 0 // The marker is rotated in one of our videos...
        glRotatef(-90, 0, 0, 1);
#endif

        shoeModel.draw();

        /* Snowman...
         glScalef(0.025F, 0.025F, 0.025F);
        // Draw 3 white spheres
        glTranslatef(0.0, 0.8, 0.0);
        drawSphere(0.8, 30, 30);
        glTranslatef(0.0, 0.8, 0.0);
        drawSphere(0.6, 30, 30);
        glTranslatef(0.0, 0.6, 0.0);
        drawSphere(0.4, 30, 30);

        // Draw the eyes
        glColor4f(0.0, 0.0, 0.0, 1.0);
        glPushMatrix();
        glTranslatef(-0.2, 0.2, 0.25);
        drawSphere(0.075, 20, 20);
        glPopMatrix();
        glPushMatrix();
        glTranslatef(0.2, 0.2, 0.25);
        drawSphere(0.075, 20, 20);
        glPopMatrix();

        glColor4f(1.0, 0.5, 0.0, 1.0);
        glTranslatef(0.0, 0, 0.3);
        drawCone(0.075, 0.4, 20, 20);
        */

        glPopMatrix();
    }

    // Flags
    for (int i = 0; i < coins.size() && i < 2; ++i)
    {
        const Coin& c = coins[i];
        const Vec3f& v = c.pos3D;

        glPushMatrix();
        glColor4f(1.0, 1.0, 1.0, 1.0);

        glTranslatef(v[0], v[1], v[2]);

        float randomizeOffset = c.id % 10;
         // drawFlag
        {
            // Base
            drawCylinder(0.01, 0.001, 20);
            glColor4f(1.0, 0.5, 0.0, 1.0);

            // Pole
            drawCylinder(0.001, 0.05, 20);

            // Some grass 
            {
                float radi = 0.01;
                float width = 0.003;
                float height = 0.01;
                glBegin(GL_TRIANGLES);
                glNormal3f(0, 0, 1);

                for (int j = 0; j < 20; j++)
                {
                    float ang = j * 91421.523542 + randomizeOffset * 4.41241; // Pseudo-random
                    float c = cos(ang);
                    float s = sin(ang);
                    float off1 = cos(time * 5 * 0.2341 + c  + s + j * 0.12345 + randomizeOffset * 42132.412) * width * 0.75;
                    float off2 = cos(time * 5 * 0.54 + c + s + 4132.3125 + j * 0.1235 + randomizeOffset * 64232.45) * width * 0.75;
                    float r1 = radi + j  * 0.0001;
                    float r2 = r1 + 0.0025 + fmod(ang * 531.1232 + 0.3125, 1) * 0.0025;

                    glColor4f(0.0, 0.4, 0.0, 1);
                    glVertex3f(c * r1 - s * width, s * r1 + c * width, 0);
                    glVertex3f(c * r1 - s * -width, s * r1 + c * -width, 0);
                    glColor4f(0.4, 0.8, 0, 1);
                    glVertex3f(c * r2 + off1, s * r2 + off2, height);

                    float d = 0.00075; // offset to avoid z-fighting with shadowmapping
                    glColor4f(0.0, 0.4, 0.0, 1);
                    glVertex3f(c * (r1 + d) - s * -width, s * (r1 + d) + c * -width, 0);
                    glVertex3f(c * (r1 + d) - s * width, s * (r1 + d) + c * width, 0);
                    glColor4f(0.4, 0.8, 0, 1);
                    glVertex3f(c * (r2 + d) + off1, s * (r2 + d) + off2, height);
                }
                glEnd();
            }

            // Actual flag
            {
                glColor4f(1.0, 0.8, 0.0, 1.0);

                //glRotatef(v[0], 1, 0, 0);
                //glRotatef(v[1], 0, 1, 0);
                glRotatef(45 + 90 + cos(-time * 5 * 0.25 + randomizeOffset * 345614.213341) * 12, 0, 0, 1);
                glTranslatef(0, 0, 0.04f);
                glRotatef(-8, 1, 0, 0);

                glBegin(GL_QUAD_STRIP);
                float length = 0.02;
                float height = 0.02;
                for (int s = 0; s < 40; s++)
                {
                    float t = (s >= 20 ? 39 - s : s) / 20.0;
                    float x = 0.001 + length * t;
                    float y = (sin(t * 5 - time * 5 + randomizeOffset * 345614.213341f) - sin(-time * 5 + randomizeOffset * 345614.213341f)) * 0.00125 + (sin(t * 2.3 - time * 5 * 2.2 + 0.3) - sin(-time * 5 * 2.2 + 0.3)) * 0.00025;
                    float dy = cos(t * 5 - time * 5 + randomizeOffset * 345614.213341f) * 5 * 0.00125 + cos(t * 2.3 - time * 5 * 2.2 + 0.3) * 2.3 * 0.00025;
                    if (s >= 20)
                        glNormal3f(dy, -length, 0);
                    else
                        glNormal3f(-dy, length, 0);
                    float d = s >= 20 ? -0.0005 : 0.0005; // offset to avoid z-fighting with shadowmapping
                    glVertex3f(x, y + d, -height * 0.5 * (1 - t));
                    glVertex3f(x, y + d, height * 0.5 * (1 - t));
                }
                glEnd();
            }
        }

        glPopMatrix();
    }

    // Soccer ball
    if (ball && coins.size() >= 3)
    {
        const Vec3f& v = coins[2].pos3D;
        const Vec3f& v1 = ballMoving ? coinMoveStartPos : v;
        const Vec3f& v2 = (coins[0].pos3D + coins[1].pos3D) * 0.5;

        glPushMatrix();
        glColor4f(1.0, 1.0, 1.0, 1.0);

        glTranslatef(v[0], v[1], v[2]);
        glScalef(0.005F, 0.005F, 0.005F);
        glTranslatef(0, 0, 2.4F);
        glRotatef(time * (ballMoving ? 180 : 60), -(v2[1] - v1[1]), v2[0] - v1[0], 0);
        glTranslatef(0, 0, -2.4F);

        glRotatef(90, 1, 0, 0);

        soccerModel.draw();
        //drawSphere(0.01F, 20, 20);
        glPopMatrix();
    }
}

// Transparent stuff (with no shadows)
void drawOverlays(float* markerMatrix, vector<Coin>& coins, bool ball, bool marker, float time)
{
    // Line between flags
    if (coins.size() >= 2)
    {
        const Vec3f& v1 = coins[0].pos3D;
        const Vec3f& v2 = coins[1].pos3D;
        
        float width = 0.0025;
        float length = 0.01;

        float dx = v2[0] - v1[0];
        float dy = v2[1] - v1[1];
        float dist = sqrt(dx * dx + dy * dy);

        dx /= dist;
        dy /= dist;

        float offset = fmod(time, 1.0);
        int stripes = (ceil(dist / length)) * 2;

        glBegin(GL_QUAD_STRIP);
        glNormal3f(0, 0, 1);

        for (int i = 0; i < stripes && i < 10000; ++i)
        {
            float f1 = (offset * 2 - 1 + i) * length * 0.5F - width * 0.25F;
            float f2 = (offset * 2 - 1 + i) * length * 0.5F + width * 0.25F;

            glColor4f(1.0, 1.0, 1.0, i % 2 != 0 ? 0.25 : 0.0);
            glVertex3f(v1[0] - dy * width + f2 * dx, v1[1] + dx * width + f2 * dy, v1[2]);
            glVertex3f(v1[0] - dy * -width + f1 * dx, v1[1] + dx * -width + f1 * dy, v1[2]);

            glColor4f(1.0, 1.0, 1.0, i % 2 != 1 ? 0.25 : 0.0);
            glVertex3f(v1[0] - dy * width + f2 * dx, v1[1] + dx * width + f2 * dy, v1[2]);
            glVertex3f(v1[0] - dy * -width + f1 * dx, v1[1] + dx * -width + f1 * dy, v1[2]);
        }

        glEnd();
    }

    // Arrow below ball 
    if (ball && !ballMoving && coins.size() >= 3)
    {
        const Vec3f& v1 = coins[2].pos3D;
        const Vec3f& v2 = (coins[0].pos3D + coins[1].pos3D) * 0.5;

        // Make arrow rotate from ball to the mid-point between the flags
        float dx = v2[0] - v1[0];
        float dy = v2[1] - v1[1];
        float dist = sqrt(dx * dx + dy * dy);
        dx /= dist;
        dy /= dist;

        float width = 0.02;
        float height = 0.015;

        float offset = 0.02 + cos(time * 5 * 0.5) * 0.0025;

        glBegin(GL_TRIANGLE_STRIP);
        glNormal3f(0, 0, 1);
        glColor4f(1.0, 1.0, 1.0, (cos(time * 5 * 0.5) * 0.125 + 0.25));

        glVertex3f(v1[0] + dx * (offset - height * 0.5F) - dy * -width * 0.5F, v1[1] + dy * (offset - height * 0.5F) + dx * -width * 0.5F, v1[2]);
        glVertex3f(v1[0] + dx * (offset + height * 0.5F), v1[1] + dy * (offset + height * 0.5F), v1[2]);
        glVertex3f(v1[0] + dx * (offset - height * 0.25F), v1[1] + dy * (offset - height * 0.25F), v1[2]);
        glVertex3f(v1[0] + dx * (offset - height * 0.5F) - dy * width * 0.5F, v1[1] + dy * (offset - height * 0.5F) + dx * width * 0.5F, v1[2]);

        glEnd();
    }

    // Goal area
    if (goals > goalsUntilFinalArea && gameState == GameState::GAME)
    {
        double scale = 1.0 * tan(fov / 180.0 * M_PI * 0.5) * (-smoothedZ);
        float texMin = fmod(1.0 - time * 0.25F, 1.0F);
        float texMaxX = texMin + (targetAreaMax[0] - targetAreaMin[0]) * 100 * scale;
        float texMaxY = texMin + (targetAreaMax[1] - targetAreaMin[1]) * 100 * scale;

        // stripes
        glColor4f(1.0, 1.0, 1.0, 0.05);
        glEnable(GL_TEXTURE_2D);
        stripeTex.bind();
        glBegin(GL_QUADS);
        glTexCoord2f(texMin, texMaxY);
        glVertex3f(targetAreaMin[0] * scale, targetAreaMin[1] * scale, smoothedZ);
        glTexCoord2f(texMaxX, texMaxY);
        glVertex3f(targetAreaMax[0] * scale, targetAreaMin[1] * scale, smoothedZ);
        glTexCoord2f(texMaxX, texMin);
        glVertex3f(targetAreaMax[0] * scale, targetAreaMax[1] * scale, smoothedZ);
        glTexCoord2f(texMin, texMin);
        glVertex3f(targetAreaMin[0] * scale, targetAreaMax[1] * scale, smoothedZ);
        glEnd();

        // goal text
        glColor4f(1.0, 1.0, 1.0, (cos(time * 5 * 0.5) * 0.125 + 0.25));
        goalTex.bind();
        glBegin(GL_QUADS);
        float height = 0.01;
        float width = height * goalTex.getWidth() / (float)goalTex.getHeight();
        float x = (targetAreaMin[0] + targetAreaMax[0]) * 0.5 * scale;
        float y = (targetAreaMin[1] + targetAreaMax[1]) * 0.5 * scale;
        glTexCoord2f(0, 1);
        glVertex3f(x - width, y - height, smoothedZ);
        glTexCoord2f(1, 1);
        glVertex3f(x + width, y - height, smoothedZ);
        glTexCoord2f(1, 0);
        glVertex3f(x + width, y + height, smoothedZ);
        glTexCoord2f(0, 0);
        glVertex3f(x - width, y + height, smoothedZ);
        glEnd();
        glDisable(GL_TEXTURE_2D);

        // border
        width = 0.005;
        float length = 0.01 + cos(time * 5 * 0.5) * 0.0025;
        glBegin(GL_TRIANGLE_STRIP);
        glVertex3f(targetAreaMin[0] * scale, targetAreaMin[1] * scale + length, smoothedZ);
        glVertex3f(targetAreaMin[0] * scale - width, targetAreaMin[1] * scale + length, smoothedZ);
        glVertex3f(targetAreaMin[0] * scale, targetAreaMin[1] * scale, smoothedZ);
        glVertex3f(targetAreaMin[0] * scale - width, targetAreaMin[1] * scale, smoothedZ);
        glVertex3f(targetAreaMin[0] * scale, targetAreaMin[1] * scale , smoothedZ);
        glVertex3f(targetAreaMin[0] * scale, targetAreaMin[1] * scale - width, smoothedZ);
        glVertex3f(targetAreaMin[0] * scale + length, targetAreaMin[1] * scale , smoothedZ);
        glVertex3f(targetAreaMin[0] * scale + length, targetAreaMin[1] * scale - width, smoothedZ);

        glVertex3f(targetAreaMin[0] * scale + length, targetAreaMin[1] * scale - width, smoothedZ);
        glVertex3f(targetAreaMax[0] * scale - length, targetAreaMin[1] * scale, smoothedZ);

        glVertex3f(targetAreaMax[0] * scale - length, targetAreaMin[1] * scale, smoothedZ);
        glVertex3f(targetAreaMax[0] * scale - length, targetAreaMin[1] * scale - width, smoothedZ);
        glVertex3f(targetAreaMax[0] * scale, targetAreaMin[1] * scale, smoothedZ);
        glVertex3f(targetAreaMax[0] * scale, targetAreaMin[1] * scale - width, smoothedZ);
        glVertex3f(targetAreaMax[0] * scale, targetAreaMin[1] * scale, smoothedZ);
        glVertex3f(targetAreaMax[0] * scale + width, targetAreaMin[1] * scale, smoothedZ);
        glVertex3f(targetAreaMax[0] * scale, targetAreaMin[1] * scale + length, smoothedZ);
        glVertex3f(targetAreaMax[0] * scale + width, targetAreaMin[1] * scale + length, smoothedZ);

        glVertex3f(targetAreaMax[0] * scale + width, targetAreaMin[1] * scale + length, smoothedZ);
        glVertex3f(targetAreaMax[0] * scale, targetAreaMax[1] * scale - length, smoothedZ);

        glVertex3f(targetAreaMax[0] * scale, targetAreaMax[1] * scale - length, smoothedZ);
        glVertex3f(targetAreaMax[0] * scale + width, targetAreaMax[1] * scale - length, smoothedZ);
        glVertex3f(targetAreaMax[0] * scale, targetAreaMax[1] * scale, smoothedZ);
        glVertex3f(targetAreaMax[0] * scale + width, targetAreaMax[1] * scale, smoothedZ);
        glVertex3f(targetAreaMax[0] * scale, targetAreaMax[1] * scale, smoothedZ);
        glVertex3f(targetAreaMax[0] * scale, targetAreaMax[1] * scale + width, smoothedZ);
        glVertex3f(targetAreaMax[0] * scale - length, targetAreaMax[1] * scale, smoothedZ);
        glVertex3f(targetAreaMax[0] * scale - length, targetAreaMax[1] * scale + width, smoothedZ);

        glVertex3f(targetAreaMax[0] * scale - length, targetAreaMax[1] * scale + width, smoothedZ);
        glVertex3f(targetAreaMin[0] * scale + length, targetAreaMax[1] * scale, smoothedZ);

        glVertex3f(targetAreaMin[0] * scale + length, targetAreaMax[1] * scale, smoothedZ);
        glVertex3f(targetAreaMin[0] * scale + length, targetAreaMax[1] * scale + width, smoothedZ);
        glVertex3f(targetAreaMin[0] * scale, targetAreaMax[1] * scale, smoothedZ);
        glVertex3f(targetAreaMin[0] * scale, targetAreaMax[1] * scale + width, smoothedZ);
        glVertex3f(targetAreaMin[0] * scale, targetAreaMax[1] * scale, smoothedZ);
        glVertex3f(targetAreaMin[0] * scale - width, targetAreaMax[1] * scale, smoothedZ);
        glVertex3f(targetAreaMin[0] * scale, targetAreaMax[1] * scale - length, smoothedZ);
        glVertex3f(targetAreaMin[0] * scale - width, targetAreaMax[1] * scale - length, smoothedZ);
        glEnd();
    }

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Particle systems
    ballFxSystem.emitting = ballMoving;
    if (ball)
        ballFxSystem.moveToPos(coins[2].pos3D, time);
    ballFxSystem.draw(time);

    goalFxSystem.draw(time);
}

// UI (title-screen / score / goal text)
void drawUI(float time)
{
    glDisable(GL_TEXTURE_2D);

    float ratio = windowWidth / (float)windowHeight;
    float backgroundOffset = 0.5F - 0.5F * cameraWidth * windowHeight / cameraHeight / windowWidth;  // Center camera image in window
    float goalAnim = min(1.0f, (time - lastGoal) / 0.75F);
    float t = min(1.0, (time - lastGameStateChange) / 0.5);
    t = gameState == GameState::TITLE ? 1 : gameState == GameState::VICTORY ? t : framesWithoutCoins > 15 ? 1 : 1 - t;
    if (t > 0.01) // Dark-background
    {
        glBegin(GL_QUADS);
        glColor4f(0.0, 0.0, 0.0, 0.8 * t);
        glVertex2f(-ratio, -1);
        glVertex2f(ratio, -1);
        glVertex2f(ratio, 1);
        glVertex2f(-ratio, 1);
        glEnd();
    }

    // Draw vignette 
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, vignetteTexture);

    glBegin(GL_QUADS);
    glColor4f(1.0, 1.0, 1.0, 1.0);
    glTexCoord2f(0, 0);
    glVertex2f((-1 + backgroundOffset * 2) * ratio, -1);
    glTexCoord2f(1, 0);
    glVertex2f((1 - backgroundOffset * 2) * ratio, -1);
    glTexCoord2f(1, 1);
    glVertex2f((1 - backgroundOffset * 2) * ratio, 1);
    glTexCoord2f(0, 1);
    glVertex2f((-1 + backgroundOffset * 2) * ratio, 1);
    glEnd();
    glDisable(GL_TEXTURE_2D);

    glBegin(GL_QUADS);
    glColor4f(0.0, 0.0, 0.0, 1.0);
    glVertex2f(-ratio, -1);
    glVertex2f((-1 + backgroundOffset * 2) * ratio, -1);
    glVertex2f((-1 + backgroundOffset * 2) * ratio, 1);
    glVertex2f(-ratio, 1);
    glVertex2f((1 - backgroundOffset * 2) * ratio, -1);
    glVertex2f(ratio, -1);
    glVertex2f(ratio, 1);
    glVertex2f((1 - backgroundOffset * 2) * ratio, 1);
    glEnd();

    if (gameState == GameState::TITLE)
    {
        // CoinSoccAR text
        glColor4f(1.0, 1.0, 1.0, 1.0);
        titleTex.drawQuad(0, 0.7, 0.25);

        float a = cos(time * M_PI) * 0.4 + 0.6;
        glColor4f(1.0, 1.0, 1.0, a);
        if (startPos)
        {
            moveCoinTex.drawQuad(0, 0.5, 0.075);

            // Draw circles of the start-positions
            glColor4f(1.0, 1.0, 1.0, startPositions[0][2] > 0.5 ? 0 :  a);
            circleTex.drawQuad(startPositions[0][0], startPositions[0][1], 0.075);
            glColor4f(1.0, 1.0, 1.0, startPositions[1][2] > 0.5 ? 0 : a);
            circleTex.drawQuad(startPositions[1][0], startPositions[1][1], 0.075);
            glColor4f(1.0, 1.0, 1.0, startPositions[2][2] > 0.5 ? 0 : a);
            circleTex.drawQuad(startPositions[2][0], startPositions[2][1], 0.075);

        }
        else
        {
            anyKeyTex.drawQuad(0, 0.5, 0.075);
        }

        glColor4f(1.0, 1.0, 1.0, 0.5);
        creditTex.drawQuad(0, -0.9, 0.075);


        // Coin in the title-logo
        glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT2);
        glEnable(GL_LIGHT0);
        glDisable(GL_LIGHT1);
        glEnable(GL_DEPTH_TEST);

        float ang = -time * 0.5 * M_PI + M_PI * 0.5;
        GLfloat light_pos3[] = { cos(ang), 0, sin(ang), 0 };
        glLightfv(GL_LIGHT2, GL_POSITION, light_pos3);

        glPushMatrix();
        glTranslatef(-0.63, 0.7, 0);
        glRotatef(time * 90, 0, 1, 0);
        glScalef(0.13, 0.13, 0.13);
        glTranslatef(0, 0.0, -0.05);

        coinModel.draw();
        glPopMatrix();

        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHT0);
        glDisable(GL_LIGHT2);
        glEnable(GL_LIGHT1);
        glDisable(GL_LIGHTING);
    }
    else if (gameState == GameState::VICTORY)
    {
        glColor4f(1.0, 1.0, 1.0, t);
        winTex.drawQuad(0,  0.25 - 0.1F * (1 - t * t), 0.15 + 0.05 * t);

        glColor4f(1.0, 1.0, 1.0, (cos(time * M_PI) * 0.4 + 0.6) * t);
        anyKeyTex.drawQuad(0, 0, 0.075);
    }
    else
    {
        if (framesWithoutCoins > 15)
        {
            glColor4f(1.0, 1.0, 1.0, (cos(time * M_PI) * 0.4 + 0.6));
            noTrackingTex.drawQuad(0, 0, 0.25f);
        }
		
        if (goalAnim < 1.0F)
        {
            glColor4f(1.0, 1.0, 1.0, sin(goalAnim * M_PI));
            float t = 1.0F - min(1.0F, goalAnim * 2);
            goalTex.drawQuad(0, 0.25, 0.15 + 0.05 * (1.0F - t * t));
        }

		//if (goals > 0) 
        {
			glColor4f(goalAnim, 1, goalAnim, 1);
			glRasterPos2f(-ratio + 0.2F, 0.8F);
			std::string scoreText = "Score: ";
			scoreText += std::to_string(goals);
			const unsigned char* score = reinterpret_cast<const unsigned char*>(scoreText.c_str());

            glutBitmapString(GLUT_BITMAP_TIMES_ROMAN_24, score);
		}
    }

    uiSystem.emitting = gameState == GameState::VICTORY || goalAnim < 0.5;
    uiSystem.draw(time);

    if (framesWithoutMarker > 15)
    {
        glColor4f(1.0, 1.0, 1.0, (cos(time * M_PI) * 0.4 + 0.6));
        noMarkerTex.drawQuad(ratio-0.2f, 1-0.2f, 0.15f);
    }
	
    glColor4f(1.0, 1.0, 1.0, 1.0);
}


void reshape(GLFWwindow* window, int width, int height)
{
    windowWidth = width;
    windowHeight = height;
    prevFov = fov;
    float ratio = width / (float)height;

    glMatrixMode(GL_PROJECTION);

    glLoadIdentity();
    gluPerspective(fov, ratio, 0.01, 100);
    glGetFloatv(GL_PROJECTION_MATRIX, cameraProjectionMatrix);

    glLoadIdentity();
    glGetFloatv(GL_PROJECTION_MATRIX, cameraViewMatrix);

    float scale = windowHeight / (float)cameraHeight;
    glPixelZoom(scale, -scale); // Scale pixels to fill the window

    int error = glGetError();
    if (error)
        std::cout << gluErrorString(error) << " in reshape\n";
}

void onKey(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_F1 && action == GLFW_PRESS)
    {
        fullscreen = !fullscreen;
        if (fullscreen)
        {
            glfwGetWindowPos(window, &savedWPosX, &savedWPosY);
            glfwGetWindowSize(window, &savedWWidth, &savedWHeight);

            int count;
            GLFWmonitor** monitors = glfwGetMonitors(&count);
            if (count == 0)
            {
                fullscreen = false;
                cout << "Failed to switch to fullscreen, no monitors!\n";
            }
            else
            {
                const GLFWvidmode* mode = glfwGetVideoMode(monitors[0]);
                glfwSetWindowMonitor(window, monitors[0], 0, 0, mode->width, mode->height, mode->refreshRate);
            }
        }
        else
            glfwSetWindowMonitor(window, NULL, savedWPosX, savedWPosY, savedWWidth, savedWHeight, 0);
    }

    if (gameState == GameState::TITLE)
    {
        if (key == GLFW_KEY_ENTER && action == GLFW_PRESS)
        {
            lastGameStateChange = glfwGetTime();
            gameState = GameState::GAME;
        }
    }
    else if (gameState == GameState::GAME)
    {
        if (key == GLFW_KEY_ENTER && action == GLFW_PRESS)
        {
            lastGameStateChange = glfwGetTime();
            gameState = GameState::VICTORY;
        }
    }
    else if (gameState == GameState::VICTORY)
    {
        if (key == GLFW_KEY_ENTER && action == GLFW_PRESS)
        {
            lastGameStateChange = glfwGetTime();
            gameState = GameState::TITLE;
            goals = 0;
        }
    }
}
