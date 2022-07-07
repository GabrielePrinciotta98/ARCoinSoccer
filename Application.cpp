#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <opencv\cv.hpp>

#include "DrawPrimitives.h"
#include "MarkerTracker.h"
#include "CoinTracker.h"
#include "PoseEstimation.h"
#include "ObjLoader.h"

#include <windows.h>
#include <mmsystem.h>

//#define CAMERA

using namespace cv;
using namespace std;

//callback for resizing the window
void reshape(GLFWwindow* window, int width, int height);
//openGL initialization
void initGL(int argc, char* argv[]);
//rendering
void drawScene(float* markerMatrix, vector<Coin>& coins, bool ball, bool marker, float time);
void drawOverlays(float* markerMatrix, vector<Coin>& coins, bool ball, bool marker, float time);

void display(GLFWwindow* window, Mat &img_background, float* markerMatrix, vector<Coin>& coins, bool ball, bool marker);

#ifdef CAMERA
const double fov = 60;
const double downscale = 0.5;
#else
const double fov = 65;
const double downscale = 0.25;
#endif

int cameraWidth, cameraHeight;
int windowWidth, windowHeight;

const GLsizei shadowMapSize = 1024;
GLuint shadowMapTexture, shadowFrameBuffer;
const bool shadowMapping = true;
float lightProjectionMatrix[16], lightViewMatrix[16], textureMatrix[16];
float cameraProjectionMatrix[16], cameraViewMatrix[16];
float smoothedZ = 100;

GLuint vignetteTexture;
ObjModel soccerModel("soccer_ball.obj", "soccer_ball_diffuse.png");
ObjModel shoeModel("football_boots.obj", "");

int main(int argc, char* argv[])
{
    //Set up OpenCV
#ifdef CAMERA
    //cv::VideoCapture cap(1);
    cv::VideoCapture cap("CoinSoccerMovie3.mp4");
#else
    cv::VideoCapture cap("CoinSoccerMovie.mp4");
#endif

    if (!cap.isOpened())
    {
        std::cout << "No capture" << std::endl;
        return -1;
    }

    GLFWwindow* window;

    if (!glfwInit())
        return -1;

    glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);
    glfwWindowHint(GLFW_SAMPLES, 8);
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
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (glewInit() != GLEW_OK)
    {
        glfwTerminate();
        return -1;
    }

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    reshape(window, width, height);

    initGL(argc, argv);

    const double kMarkerSize = 0.045;
    MarkerTracker markerTracker(kMarkerSize, 120, 120);
#ifdef CAMERA
    CoinTracker coinTracker(11 * downscale);
#else
    CoinTracker coinTracker(24 * downscale);
#endif
    std::vector<Coin> lastCoins;
    std::vector<Coin> coins;
    cv::Vec2f lastBallPosition(0, 0);
    int framesWithoutMovement = 60;
    size_t ballID = -1;
    bool ballMoving = false;

    cv::Mat img_background, img_background_resized, img_background2;
    float resultMatrix[16];
    while (!glfwWindowShouldClose(window))
    {
        if (!cap.read(img_background))
        {
            cap.set(CAP_PROP_POS_FRAMES, 0); // Loop back to frame 0
            if (!cap.read(img_background))
                break;
        }

        img_background2 = img_background.clone();
        lastCoins = coins;

        cv::resize(img_background, img_background_resized, Size(), downscale, downscale, CV_INTER_LINEAR);
        coinTracker.findCoins(img_background_resized, lastCoins, coins);
        bool marker = markerTracker.findMarker(img_background, resultMatrix);
        
        // xyz Position from last markerMatrix
        double x = resultMatrix[3];
        double y = resultMatrix[7];
        double z = resultMatrix[11];
        smoothedZ = abs(smoothedZ - z) > 1 ? z : smoothedZ * 0.9 + z * 0.1;
        // For mapping 2d pixel coordinates to 3d coordinates, adjust for (downscaled) resolution and perspective
        double scale = 2.0 * tan(fov / 180.0 * M_PI * 0.5) * (-smoothedZ) / ((double)img_background.rows * downscale);

        float closestDist = 0;
        int closest = -1;
        float fastestVel = 0;
        int fastest = -1;

        if (!ballMoving && coins.size() > 3)
            coins.resize(3);

        for (int i = 0; i < coins.size(); ++i)
        {
            Coin& c = coins[i];
            c.pos3D[0] = (c.pos2D[0] - img_background.cols * downscale * 0.5) * scale;
            c.pos3D[1] = (img_background.rows * downscale * 0.5 - c.pos2D[1]) * scale;
            c.pos3D[2] = smoothedZ;

            float vel = c.vel[2];
            if (fastest == -1 || fastestVel < vel)
            {
                fastestVel = vel;
                fastest = i;
            }

            float dx = c.pos3D[0] - x;
            float dy = c.pos3D[1] - y;
            float dist = dx * dx + dy * dy;
            if ((closest == -1 || dist < closestDist) && c.framesTracked > 3)
            {
                closestDist = dist;
                closest = i;
            }
        }

        int ball = -1;
        if (coins.size() >= 3)
        {
            if (fastestVel > 6)
            {
                ball = fastest;
                framesWithoutMovement = 0;
                ballMoving = true;
            }
            else
            {
                if (framesWithoutMovement > 5 && coins[0].framesTracked > 3 && coins[1].framesTracked > 3)
                {
                    ballMoving = false;
                    ball = closest;
                }
                else
                {
                    if (coins[2].id == ballID)
                        ball = 2;
                    else if (coins[1].id == ballID)
                        ball = 1;
                    else if (coins[0].id == ballID)
                        ball = 0;
                    else
                    {
                        closest = -1;
                        for (int i = 0; i < coins.size(); ++i)
                        {
                            if (coins[i].framesTracked == 0)
                            {
                                float dx = coins[i].pos2D[0] - lastBallPosition[0];
                                float dy = coins[i].pos2D[1] - lastBallPosition[1];
                                float dist = dx * dx + dy * dy;
                                if ((closest == -1 || dist < closestDist) && dist < coinTracker.coinSize * coinTracker.coinSize * 3 * 3)
                                {
                                    closestDist = dist;
                                    closest = i;
                                }
                            }
                        }
                        ball = closest;
                        ballMoving = true;
                    }
                }

                framesWithoutMovement++;
            }
            if (ball != -1 && ball != 2)
            {
                swap(coins[2], coins[ball]);
                ball = 2;
            }

            if (ball != -1)
            {
                ballID = coins[ball].id;
                const Vec2f& ballPos = coins[ball].pos2D;
                const Vec2f& pos1 = coins[0].pos2D;
                const Vec2f& pos2 = coins[1].pos2D;

                if (ballMoving)
                {
                    float lastDist = (pos2[1] - lastBallPosition[1]) * (pos2[0] - pos1[0]) - (pos2[0] - lastBallPosition[0]) * (pos2[1] - pos1[1]);
                    float dist = (pos2[1] - ballPos[1]) * (pos2[0] - pos1[0]) - (pos2[0] - ballPos[0]) * (pos2[1] - pos1[1]);

                    if (signbit(lastDist) != signbit(dist))
                    {
                        cout << "Goal" << "\n";
                        PlaySound(TEXT("goal.wav"), NULL, SND_FILENAME | SND_ASYNC);

                        Point start(cvRound(pos1[0] / downscale), cvRound(pos1[1] / downscale));
                        Point end(cvRound(pos2[0] / downscale), cvRound(pos2[1] / downscale));
                        line(img_background2, start, end, Scalar(0, 0, 255), 5);
                    }
                }

                lastBallPosition = ballPos;
            }
        }
        else
        {
            if (coins.size() == 2 && coins[0].framesTracked > 3 && coins[1].framesTracked > 3)
                ballMoving = true;

            framesWithoutMovement = 0;
        }
        if (ballMoving)
        {
            Point start(100, 100);
            Point end(100, 200);
            line(img_background2, start, end, Scalar(0, 0, 255), 5);
        }

        display(window, img_background2, resultMatrix, coins, ball != -1, marker);

        glfwSwapBuffers(window);
        glfwPollEvents();

        int key = cv::waitKey(1);
        if (key == 27)
            break;
    }

    glfwTerminate();
    return 0;
}


void initGL(int argc, char* argv[])
{

    GLfloat zero[] = { 0, 0, 0, 1.0 };
    GLfloat one[] = { 1.0, 1.0, 1.0, 1.0 };
    GLfloat light_pos[] = { 0.5, 0.3, 1.0, 0 };
    GLfloat light_pos2[] = { -0.5, -0.3, -1.0, 0 };
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

    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, light_spe);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 120.0);
    glEnable(GL_LIGHT1);

    glLightModelf(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE);
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, zero);

    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_NORMALIZE);
    glEnable(GL_DITHER);
    glEnable(GL_FRAMEBUFFER_SRGB);
    glEnable(GL_CULL_FACE);

    glClearColor(0., 0., 0., 1);
    glClearDepth(1);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_DEPTH_TEST);

    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // Shadowmapping with fixed-pipeline based on http://www.paulsprojects.net/tutorials/smt/smt.html
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
    glOrtho(-0.25, 0.25, -0.25, 0.25, 1, 1.5);
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

    soccerModel.init();
    shoeModel.init();

    int error = glGetError();
    if (error)
        std::cout << gluErrorString(error) << " in init\n";
}

void display(GLFWwindow* window, Mat &img_background, float* markerMatrix, vector<Coin>& coins, bool ball, bool marker)
{
    float time = glfwGetTime() * 5;

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

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(-1.0f, 1.0f, -1.0f, 1.0f);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, vignetteTexture);
    glBegin(GL_QUADS);
    glColor4f(1.0, 1.0, 1.0, 0.8);
    glTexCoord2f(0, 0);
    glVertex2f(-1 + backgroundOffset * 2, -1);
    glTexCoord2f(1, 0);
    glVertex2f(1 - backgroundOffset * 2, -1);
    glTexCoord2f(1, 1);
    glVertex2f(1 - backgroundOffset * 2, 1);
    glTexCoord2f(0, 1);
    glVertex2f(-1 + backgroundOffset * 2, 1);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glDisable(GL_BLEND);

    int error = glGetError();
    if (error)
        std::cout << gluErrorString(error) << " in display\n";
}

void drawScene(float* markerMatrix, vector<Coin>& coins, bool ball, bool marker, float time)
{
    if (marker)
    {
        glPushMatrix();
        glMultTransposeMatrixf(markerMatrix);
        glColor4f(1.0, 1.0, 1.0, 1.0);

        glScalef(0.00075F, 0.00075F, 0.00075F);
        glRotatef(-90, 1, 0, 0);
        glTranslatef(0, 0, -20.F);
        glRotatef(-90, 0, 1, 0);

        shoeModel.draw();

        /*
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
            drawCylinder(0.01, 0.001, 20);
            glColor4f(1.0, 0.5, 0.0, 1.0);

            drawCylinder(0.001, 0.05, 20);

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
                    float off1 = cos(time * 0.2341 + c  + s + j * 0.12345 + randomizeOffset * 42132.412) * width * 0.75;
                    float off2 = cos(time * 0.54 + c + s + 4132.3125 + j * 0.1235 + randomizeOffset * 64232.45) * width * 0.75;
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

            {
                glColor4f(1.0, 0.8, 0.0, 1.0);

                //glRotatef(v[0], 1, 0, 0);
                //glRotatef(v[1], 0, 1, 0);
                glRotatef(45 + 90 + cos(-time * 0.25 + randomizeOffset * 345614.213341) * 12, 0, 0, 1);
                glTranslatef(0, 0, 0.04f);
                glRotatef(-8, 1, 0, 0);

                glBegin(GL_QUAD_STRIP);
                float length = 0.02;
                float height = 0.02;
                for (int s = 0; s < 40; s++)
                {
                    float t = (s >= 20 ? 39 - s : s) / 20.0;
                    float x = 0.001 + length * t;
                    float y = (sin(t * 5 - time + randomizeOffset * 345614.213341f) - sin(-time + randomizeOffset * 345614.213341f)) * 0.00125 + (sin(t * 2.3 - time * 2.2 + 0.3) - sin(-time * 2.2 + 0.3)) * 0.00025;
                    float dy = cos(t * 5 - time + randomizeOffset * 345614.213341f) * 5 * 0.00125 + cos(t * 2.3 - time * 2.2 + 0.3) * 2.3 * 0.00025;
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
        const Coin& c = coins[2];
        const Vec3f& v = c.pos3D;

        glPushMatrix();
        glColor4f(1.0, 1.0, 1.0, 1.0);

        glTranslatef(v[0], v[1], v[2]);
        glScalef(0.005F, 0.005F, 0.005F);
        glTranslatef(0, 0, 2.4F);
        glRotatef(time * 10, 1, 0, 0);
        glTranslatef(0, 0, -2.4F);

        glRotatef(90, 1, 0, 0);

        soccerModel.draw();
        //drawSphere(0.01F, 20, 20);
        glPopMatrix();
    }
}

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

        float offset = fmod(time / 5, 1.0);

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
    if (ball)
    {
        const Vec3f& v1 = coins[2].pos3D;
        const Vec3f& v2 = (coins[0].pos3D + coins[1].pos3D) * 0.5;

        float dx = v2[0] - v1[0];
        float dy = v2[1] - v1[1];
        float dist = sqrt(dx * dx + dy * dy);

        dx /= dist;
        dy /= dist;

        float width = 0.02;
        float height = 0.015;

        float offset = 0.02 + cos(time * 0.5) * 0.0025;

        glBegin(GL_TRIANGLE_STRIP);
        glNormal3f(0, 0, 1);
        glColor4f(1.0, 1.0, 1.0, (cos(time * 0.5) * 0.125 + 0.25));

        glVertex3f(v1[0] + dx * (offset - height * 0.5F) - dy * -width * 0.5F, v1[1] + dy * (offset - height * 0.5F) + dx * -width * 0.5F, v1[2]);
        glVertex3f(v1[0] + dx * (offset + height * 0.5F), v1[1] + dy * (offset + height * 0.5F), v1[2]);
        glVertex3f(v1[0] + dx * (offset - height * 0.25F), v1[1] + dy * (offset - height * 0.25F), v1[2]);
        glVertex3f(v1[0] + dx * (offset - height * 0.5F) - dy * width * 0.5F, v1[1] + dy * (offset - height * 0.5F) + dx * width * 0.5F, v1[2]);

        glEnd();
    }
}

void reshape(GLFWwindow* window, int width, int height)
{
    windowWidth = width;
    windowHeight = height;
    glViewport(0, 0, (GLsizei)width, (GLsizei)height);
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