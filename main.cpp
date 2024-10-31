#include <iostream>
#include <vector>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>


#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#ifndef MAX_PATH
#define MAX_PATH 260
#endif


using namespace std;
GLFWwindow* window;
using namespace std;
const int width = 1024, height = 1024;
GLuint VAO, VBO, EBO;
GLuint shaderProgram;

int numX = 20, numY = 20;
const size_t total_points = (numX + 1) * (numY + 1);
float fullsize = 4.0f;
float halfsize = fullsize / 2.0f;

char info[MAX_PATH] = {0};

int selected_index = -1;

struct Spring
{ // models the springs between points of the cloth
	int p1, p2;
	float rest_length;
	float Ks, Kd;
	int type;
};

const float EPS = 0.001f;
const float EPS2 = EPS * EPS;
const int i_max = 10;

glm::mat4 ellipsoid, inverse_ellipsoid;
int iStacks = 30;
int iSlices = 30;
float fRadius = 1;

// Resolve constraint in object space
glm::vec3 center = glm::vec3(0, 0, 0); // object space center of ellipsoid
float radius = 1;					   // object space radius of ellipsoid

std::chrono::high_resolution_clock::time_point startTimePoint, endTimePoint;
double frameTimeQP = 0;
float frameTime = 0;
float fps = 0;
float startTime = 0;
int totalFrames = 0;


GLuint LoadShaders(const char *vertex_file_path, const char *fragment_file_path) {
    // Create shaders
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    // Read the Vertex Shader code from file
    std::string vertexCode;
    std::ifstream vertexFile(vertex_file_path);
    if (vertexFile) {
        std::stringstream vertexStream;
        vertexStream << vertexFile.rdbuf();
        vertexCode = vertexStream.str();
        vertexFile.close();
    } else {
        std::cerr << "Failed to open vertex shader file: " << vertex_file_path << std::endl;
        return 0;
    }

    // Read the Fragment Shader code from file
    std::string fragmentCode;
    std::ifstream fragmentFile(fragment_file_path);
    if (fragmentFile) {
        std::stringstream fragmentStream;
        fragmentStream << fragmentFile.rdbuf();
        fragmentCode = fragmentStream.str();
        fragmentFile.close();
    } else {
        std::cerr << "Failed to open fragment shader file: " << fragment_file_path << std::endl;
        return 0;
    }

    // Compile Vertex Shader
    const char* vertexSource = vertexCode.c_str();
    glShaderSource(vertexShader, 1, &vertexSource, NULL);
    glCompileShader(vertexShader);

    // Check Vertex Shader compilation
    GLint success;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cerr << "Vertex Shader compilation failed:\n" << infoLog << std::endl;
        return 0;
    }

    // Compile Fragment Shader
    const char* fragmentSource = fragmentCode.c_str();
    glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
    glCompileShader(fragmentShader);

    // Check Fragment Shader compilation
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cerr << "Fragment Shader compilation failed:\n" << infoLog << std::endl;
        return 0;
    }

    // Link shaders into a shader program
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // Check the linking
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cerr << "Shader Program linking failed:\n" << infoLog << std::endl;
        return 0;
    }

    // Clean up shaders bc theyre linked into the program now and no longer needed
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}

void startTiming(){
	startTimePoint = std::chrono::high_resolution_clock::now();
}

void endTiming(){
	endTimePoint = std::chrono::high_resolution_clock::now();
	frameTimeQP = std::chrono::duration<double, std::milli>(endTimePoint - startTimePoint).count();
}

void updateWindowTitle(float fps, float frameTime){
	snprintf(info, sizeof(info), "FPS: %3.2f, Frame time (GLUT): %3.4f msecs, Frame time (QP): %3.3f", fps, frameTime, frameTimeQP);
	glfwSetWindowTitle(window, info);
}



void SolveConjugateGradient2(glm::mat2 A, glm::vec2 &x, glm::vec2 b)
{
	float i = 0;
	glm::vec2 r = b - A * x;
	glm::vec2 d = r;
	glm::vec2 q = glm::vec2(0); // initializes both components of 'q' to 0
	float alpha_new = 0;
	float alpha = 0;
	float beta = 0;
	float delta_old = 0;
	float delta_new = glm::dot(r, r);
	float delta0 = delta_new; // what is this for?
	while (i < i_max && delta_new > EPS2)
	{
		q = A * d;
		alpha = delta_new / glm::dot(d, q);
		x = x + alpha * d;
		r = r - alpha * q;
		delta_old = delta_new;
		delta_new = glm::dot(r, r);
		beta = delta_new / delta_old;
		d = r + beta * d;
		i++;
	}
}

template <class T> // defines LargeVector class that can hold a vector of any data type T
class LargeVector
{
private:
	vector<T> v;

public:
	LargeVector()
	{ // default constructor
	}
	LargeVector(const LargeVector &other)
	{																	   // Copy Constructor: copies the contents of other
		v.resize(other.v.size());										   // by resizing the vector v into the size of other
		memcpy(&v[0], &(other.v[0]), sizeof(other.v[0]) * other.v.size()); // and copies the data from other into v
	}
	void resize(const int size)
	{
		v.resize(size);
	}
	void clear(bool isIdentity = false)
	{
		memset(&v[0], 0, sizeof(T) * v.size());
		if (isIdentity)
		{
			for (size_t i = 0; i < v.size(); i++)
			{
				v[i] = T(1);
			}
		}
	}
	size_t size()
	{ // returns the size of the vector
		return v.size();
	}

	T &operator[](int index)
	{ // allows indexing, returns v[i]
		return v[index];
	}

	// Friend Functions: are allowed access to private data memebers of the class
	//  these friend functions are specifically deisgned for 3d vectors (glm::vec3) and 3x3 matrices (glm::mat3)
	friend LargeVector<glm::vec3> operator*(const LargeVector<glm::mat3> other, const LargeVector<glm::vec3> f);
	friend LargeVector<glm::vec3> operator*(const float f, const LargeVector<glm::vec3> other);
	friend LargeVector<glm::vec3> operator-(const LargeVector<glm::vec3> Va, const LargeVector<glm::vec3> Vb);
	// friend LargeVector<T> operator+(const LargeVector<T> Va, const LargeVector<T> Vb );
	friend LargeVector<glm::vec3> operator+(const LargeVector<glm::vec3> Va, const LargeVector<glm::vec3> Vb);

	friend LargeVector<glm::mat3> operator*(const float f, const LargeVector<glm::mat3> other);
	friend LargeVector<glm::mat3> operator-(const LargeVector<glm::mat3> Va, const LargeVector<glm::mat3> Vb);
	// friend LargeVector<glm::mat3> operator+(const LargeVector<glm::mat3> Va, const LargeVector<glm::mat3> Vb );

	friend LargeVector<glm::vec3> operator/(const float f, const LargeVector<glm::vec3> v);
	friend float dot(const LargeVector<glm::vec3> Va, const LargeVector<glm::vec3> Vb);
};

// these functions cannot perform operations on private data memebers because they do not belong to a class
LargeVector<glm::vec3> operator*(const LargeVector<glm::mat3> other, const LargeVector<glm::vec3> v)
{
	LargeVector<glm::vec3> tmp(v);
	for (size_t i = 0; i < v.v.size(); i++)
	{
		tmp.v[i] = other.v[i] * v.v[i];
	}
	return tmp;
}

LargeVector<glm::vec3> operator*(const float f, const LargeVector<glm::vec3> other)
{
	LargeVector<glm::vec3> tmp(other);
	for (size_t i = 0; i < other.v.size(); i++)
	{
		tmp.v[i] = other.v[i] * f;
	}
	return tmp;
}
LargeVector<glm::mat3> operator*(const float f, const LargeVector<glm::mat3> other)
{
	LargeVector<glm::mat3> tmp(other);
	for (size_t i = 0; i < other.v.size(); i++)
	{
		tmp.v[i] = other.v[i] * f;
	}
	return tmp;
}
LargeVector<glm::vec3> operator-(const LargeVector<glm::vec3> Va, const LargeVector<glm::vec3> Vb)
{
	LargeVector<glm::vec3> tmp(Va);
	for (size_t i = 0; i < Va.v.size(); i++)
	{
		tmp.v[i] = Va.v[i] - Vb.v[i];
	}
	return tmp;
}
LargeVector<glm::mat3> operator-(const LargeVector<glm::mat3> Va, const LargeVector<glm::mat3> Vb)
{
	LargeVector<glm::mat3> tmp(Va);
	for (size_t i = 0; i < Va.v.size(); i++)
	{
		tmp.v[i] = Va.v[i] - Vb.v[i];
	}
	return tmp;
}

LargeVector<glm::vec3> operator+(const LargeVector<glm::vec3> Va, const LargeVector<glm::vec3> Vb)
{
	LargeVector<glm::vec3> tmp(Va);
	for (size_t i = 0; i < Va.v.size(); i++)
	{
		tmp.v[i] = Va.v[i] + Vb.v[i];
	}
	return tmp;
}

LargeVector<glm::vec3> operator/(const float f, const LargeVector<glm::vec3> v)
{
	LargeVector<glm::vec3> tmp(v);
	for (size_t i = 0; i < v.v.size(); i++)
	{
		tmp.v[i] = v.v[i] / f;
	}
	return tmp;
}

float dot(const LargeVector<glm::vec3> Va, const LargeVector<glm::vec3> Vb)
{
	float sum = 0;
	for (size_t i = 0; i < Va.v.size(); i++)
	{
		sum += glm::dot(Va.v[i], Vb.v[i]);
	}
	return sum;
}

void SolveConjugateGradient(LargeVector<glm::mat3> A, LargeVector<glm::vec3> &x, LargeVector<glm::vec3> b)
{
	float i = 0;
	LargeVector<glm::vec3> r = b - A * x;
	LargeVector<glm::vec3> d = r;
	LargeVector<glm::vec3> q;
	float alpha_new = 0;
	float alpha = 0;
	float beta = 0;
	float delta_old = 0;
	float delta_new = dot(r, r);
	float delta0 = delta_new;
	while (i < i_max && delta_new > EPS2)
	{
		q = A * d;
		alpha = delta_new / dot(d, q);
		x = x + alpha * d;
		r = r - alpha * q;
		delta_old = delta_new;
		delta_new = dot(r, r);
		beta = delta_new / delta_old;
		d = r + beta * d;
		i++;
	}
}

void SolveConjugateGradient(glm::mat3 A, glm::vec3 &x, glm::vec3 b)
{
	float i = 0;
	glm::vec3 r = b - A * x;
	glm::vec3 d = r;
	glm::vec3 q = glm::vec3(0);
	float alpha_new = 0;
	float alpha = 0;
	float beta = 0;
	float delta_old = 0;
	float delta_new = glm::dot(r, r);
	float delta0 = delta_new;
	while (i < i_max && delta_new > EPS2)
	{
		q = A * d;
		alpha = delta_new / glm::dot(d, q);
		x = x + alpha * d;
		r = r - alpha * q;
		delta_old = delta_new;
		delta_new = glm::dot(r, r);
		beta = delta_new / delta_old;
		d = r + beta * d;
		i++;
	}
}

vector<GLushort> indices;
vector<Spring> springs;

LargeVector<glm::vec3> X;
LargeVector<glm::vec3> V;
LargeVector<glm::vec3> F;

LargeVector<glm::mat3> df_dx; //  df/dp
LargeVector<glm::vec3> dc_dp; //  df/dp

vector<glm::vec3> deltaP2;
LargeVector<glm::vec3> V_new;
LargeVector<glm::mat3> M;	// the mass matrix
glm::mat3 I = glm::mat3(1); // identity matrix

LargeVector<glm::mat3> K; // stiffness matrix

vector<float> C;	 // for implicit integration
vector<float> C_Dot; // for implicit integration

int oldX = 0, oldY = 0;
float rX = 15, rY = 0;
int state = 1;
float dist = -23;
const int GRID_SIZE = 10;

const int STRUCTURAL_SPRING = 0;
const int SHEAR_SPRING = 1;
const int BEND_SPRING = 2;
int spring_count = 0;

const float DEFAULT_DAMPING = -0.0125f;
float KsStruct = 0.75f, KdStruct = -0.25f;
float KsShear = 0.75f, KdShear = -0.25f;
float KsBend = 0.95f, KdBend = -0.25f;
glm::vec3 gravity = glm::vec3(0.0f, -0.00981f, 0.0f);
float mass = 0.5f;

float timeStep = 1 / 60.0f;
float currentTime = 0;
double accumulator = timeStep;


GLint viewport[4];
GLdouble MV[16];
GLdouble P[16];

glm::vec3 Up = glm::vec3(0, 1, 0), Right, viewDir;

void AddSpring(int a, int b, float ks, float kd, int type)
{
	Spring spring;
	spring.p1 = a;
	spring.p2 = b;
	spring.Ks = ks;
	spring.Kd = kd;
	spring.type = type;
	glm::vec3 deltaP = X[a] - X[b];
	spring.rest_length = sqrt(glm::dot(deltaP, deltaP));
	springs.push_back(spring);

}

void OnMouseDown(int button, int action, int xpos, int ypos) {
    if (action == GLFW_PRESS) {
        // Handle mouse press
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            // Left button pressed
        } else if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
            // Middle button pressed
        } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            // Right button pressed
        }
    } else if (action == GLFW_RELEASE) {
        // Handle mouse release
    }
}
void OnMouseButton(GLFWwindow *window, int button, int action, int mods) {
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    OnMouseDown(button, action, (int)xpos, (int)ypos);
}



void OnMouseMove(int x, int y)
{
	if (selected_index == -1)
	{
		if (state == 0)
			dist *= (1 + (y - oldY) / 60.0f);
		else
		{
			rY += (x - oldX) / 5.0f;
			rX += (y - oldY) / 5.0f;
		}
	}
	else
	{
		float delta = 1500 / abs(dist);
		float valX = (x - oldX) / delta;
		float valY = (oldY - y) / delta;
		if (abs(valX) > abs(valY))
			// glutSetCursor(GLUT_CURSOR_LEFT_RIGHT);
			//  GLFW doesn't have direct equivalents for all GLUT cursors
			//  might need to hide or show the cursor
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL); // or GLFW_CURSOR_DISABLED
		else
			// glutSetCursor(GLUT_CURSOR_UP_DOWN);
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

		V[selected_index] = glm::vec3(0);
		X[selected_index].x += Right[0] * valX;
		float newValue = X[selected_index].y + Up[1] * valY;
		if (newValue > 0)
			X[selected_index].y = newValue;
		X[selected_index].z += Right[2] * valX + Up[2] * valY;
	}
	oldX = x;
	oldY = y;
}
void OnCursorPos(GLFWwindow *window, double xpos, double ypos){
	OnMouseMove((int)xpos, (int)ypos);
}

void DrawGrid(){
	glBegin(GL_LINES);
	glColor3f(0.5f, 0.5f, 0.5f);
	for (int i = -GRID_SIZE; i <= GRID_SIZE; i++){
		glVertex3f((float)i, 0, (float)-GRID_SIZE);
		glVertex3f((float)i, 0, (float)GRID_SIZE);
		glVertex3f((float)-GRID_SIZE, 0, (float)i);
		glVertex3f((float)GRID_SIZE, 0, (float)i);
	}
	glEnd();
}

void InitGL(){
	startTime = glfwGetTime();
	currentTime = startTime;

	// Generate and bind the VAO (Vertex Array Object)
	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	// Generate and bind the VBO (Vertex Buffer Object) for vertex positions
	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	// Upload the vertex data to the GPU
	glBufferData(GL_ARRAY_BUFFER, X.size() * sizeof(glm::vec3), &X[0], GL_DYNAMIC_DRAW);

	// Specify the layout of the vertex data
	// Assuming your vertex shader has 'layout(location = 0) in vec3 aPos;'
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);
	glEnableVertexAttribArray(0);

	// Unbind the VAO to prevent accidental modification
	glBindVertexArray(0);

	glEnable(GL_DEPTH_TEST);
	int i = 0, j = 0, count = 0;
	int l1 = 0, l2 = 0;
	int v = numY + 1;
	int u = numX + 1;

	indices.resize(numX * numY * 2 * 3);
	X.resize(total_points);
	V.resize(total_points);
	F.resize(total_points);
	V_new.resize(total_points);

	for (j = 0; j <= numY; j++)
	{
		for (i = 0; i <= numX; i++)
		{
			X[count++] = glm::vec3(((float(i) / (u - 1)) * 2 - 1) * halfsize, fullsize + 1, ((float(j) / (v - 1)) * fullsize));
		}
	}

	memset(&(V[0].x), 0, total_points * sizeof(glm::vec3));

	// Initialize indices for drawing triangles
	GLushort *id = &indices[0];
	for (i = 0; i < numY; i++)
	{
		for (j = 0; j < numX; j++)
		{
			int i0 = i * (numX + 1) + j;
			int i1 = i0 + 1;
			int i2 = i0 + (numX + 1);
			int i3 = i2 + 1;
			if ((j + i) % 2)
			{
				*id++ = i0;
				*id++ = i2;
				*id++ = i1;
				*id++ = i1;
				*id++ = i2;
				*id++ = i3;
			}
			else
			{
				*id++ = i0;
				*id++ = i2;
				*id++ = i3;
				*id++ = i0;
				*id++ = i3;
				*id++ = i1;
			}
		}
	}

	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glPointSize(5);

	// Setup springs for cloth simulation
	for (l1 = 0; l1 < v; l1++)
		for (l2 = 0; l2 < (u - 1); l2++)
		{
			AddSpring((l1 * u) + l2, (l1 * u) + l2 + 1, KsStruct, KdStruct, STRUCTURAL_SPRING);
		}

	// Vertical springs
	for (l1 = 0; l1 < (u); l1++)
		for (l2 = 0; l2 < (v - 1); l2++)
		{
			AddSpring((l2 * u) + l1, ((l2 + 1) * u) + l1, KsStruct, KdStruct, STRUCTURAL_SPRING);
		}

	// Shearing springs
	for (l1 = 0; l1 < (v - 1); l1++)
		for (l2 = 0; l2 < (u - 1); l2++)
		{
			AddSpring((l1 * u) + l2, ((l1 + 1) * u) + l2 + 1, KsShear, KdShear, SHEAR_SPRING);
			AddSpring(((l1 + 1) * u) + l2, (l1 * u) + l2 + 1, KsShear, KdShear, SHEAR_SPRING);
		}

	// Bend springs
	for (l1 = 0; l1 < (v); l1++)
	{
		for (l2 = 0; l2 < (u - 2); l2++)
		{
			AddSpring((l1 * u) + l2, (l1 * u) + l2 + 2, KsBend, KdBend, BEND_SPRING);
		}
		AddSpring((l1 * u) + (u - 3), (l1 * u) + (u - 1), KsBend, KdBend, BEND_SPRING);
	}
	for (l1 = 0; l1 < (u); l1++)
	{
		for (l2 = 0; l2 < (v - 2); l2++)
		{
			AddSpring((l2 * u) + l1, ((l2 + 2) * u) + l1, KsBend, KdBend, BEND_SPRING);
		}
		AddSpring(((v - 3) * u) + l1, ((v - 1) * u) + l1, KsBend, KdBend, BEND_SPRING);
	}
	int total_springs = springs.size();

	M.resize(total_springs);
	M = mass * M;
	K.resize(total_springs);

	// Create basic ellipsoid object
	ellipsoid = glm::translate(glm::mat4(1), glm::vec3(0, 2, 0));
	ellipsoid = glm::rotate(ellipsoid, 45.0f, glm::vec3(1, 0, 0));
	ellipsoid = glm::scale(ellipsoid, glm::vec3(fRadius, fRadius, fRadius / 2));
	inverse_ellipsoid = glm::inverse(ellipsoid);
}

void OnReshape(int nw, int nh)
{
    glViewport(0, 0, nw, nh);

    // Set up a perspective matrix using GLM
    glm::mat4 projection = glm::perspective(glm::radians(60.0f), (float)nw / (float)nh, 1.0f, 100.0f);

    // Update the projection matrix (if required in shader or uniform)
    GLuint projMatrixLoc = glGetUniformLocation(shaderProgram, "projectionMatrix");
    glUniformMatrix4fv(projMatrixLoc, 1, GL_FALSE, glm::value_ptr(projection));

    // No need to use `glMatrixMode` and `gluPerspective` with this approach
}


void OnShutdown(){
	X.clear();
	V.clear();
	V_new.clear();
	M.clear();
	F.clear();
	springs.clear();
	indices.clear();
}

void ComputeForces(){
	size_t i = 0;

	for (i = 0; i < total_points; i++)
	{
		F[i] = glm::vec3(0);

		// add gravity force
		if (i != 0 && i != (numX))
			F[i] += gravity;

		// F[i] += DEFAULT_DAMPING*V[i];
	}

	for (i = 0; i < springs.size(); i++)
	{
		glm::vec3 p1 = X[springs[i].p1];
		glm::vec3 p2 = X[springs[i].p2];
		glm::vec3 v1 = V[springs[i].p1];
		glm::vec3 v2 = V[springs[i].p2];
		glm::vec3 deltaP = p1 - p2;
		glm::vec3 deltaV = v1 - v2;
		float dist = glm::length(deltaP);

		// fill in the Jacobian matrix
		float dist2 = dist * dist;
		float lo_l = springs[i].rest_length / dist;
		K[i] = springs[i].Ks * (-I + (lo_l * (I - glm::outerProduct(deltaP, deltaP) / dist2)));

		float leftTerm = -springs[i].Ks * (dist - springs[i].rest_length);
		float rightTerm = springs[i].Kd * (glm::dot(deltaV, deltaP) / dist);
		glm::vec3 springForce = (leftTerm + rightTerm) * glm::normalize(deltaP);

		if (springs[i].p1 != 0 && springs[i].p1 != numX)
			F[springs[i].p1] += springForce;
		if (springs[i].p2 != 0 && springs[i].p2 != numX)
			F[springs[i].p2] -= springForce;
	}
}

void IntegrateImplicit(float deltaTime)
{
	float deltaT2 = deltaTime * deltaTime;

	LargeVector<glm::mat3> A = M - deltaT2 * K;
	LargeVector<glm::vec3> b = M * V + deltaTime * F;

	SolveConjugateGradient(A, V_new, b);

	for (size_t i = 0; i < total_points; i++)
	{
		X[i] += deltaTime * V_new[i];
		if (X[i].y < 0)
		{
			X[i].y = 0;
		}
		V[i] = V_new[i];
	}
}

void ApplyProvotDynamicInverse(){

	for (size_t i = 0; i < springs.size(); i++)
	{
		// check the current lengths of all springs
		glm::vec3 p1 = X[springs[i].p1];
		glm::vec3 p2 = X[springs[i].p2];
		glm::vec3 deltaP = p1 - p2;

		float dist = glm::length(deltaP);
		if (dist > springs[i].rest_length)
		{
			dist -= (springs[i].rest_length);
			dist /= 2.0f;
			deltaP = glm::normalize(deltaP);
			deltaP *= dist;
			if (springs[i].p1 == 0 || springs[i].p1 == numX)
			{
				V[springs[i].p2] += deltaP;
			}
			else if (springs[i].p2 == 0 || springs[i].p2 == numX)
			{
				V[springs[i].p1] -= deltaP;
			}
			else
			{
				V[springs[i].p1] -= deltaP;
				V[springs[i].p2] += deltaP;
			}
		}
	}
}
void EllipsoidCollision()
{
	for (size_t i = 0; i < total_points; i++)
	{
		glm::vec4 X_0 = (inverse_ellipsoid * glm::vec4(X[i], 1));
		glm::vec3 delta0 = glm::vec3(X_0.x, X_0.y, X_0.z) - center;
		float distance = glm::length(delta0);
		if (distance < 1.0f)
		{
			delta0 = (radius - distance) * delta0 / distance;

			// Transform the delta back to original space
			glm::vec3 delta;
			glm::vec3 transformInv;
			transformInv = glm::vec3(ellipsoid[0].x, ellipsoid[1].x, ellipsoid[2].x);
			transformInv /= glm::dot(transformInv, transformInv);
			delta.x = glm::dot(delta0, transformInv);
			transformInv = glm::vec3(ellipsoid[0].y, ellipsoid[1].y, ellipsoid[2].y);
			transformInv /= glm::dot(transformInv, transformInv);
			delta.y = glm::dot(delta0, transformInv);
			transformInv = glm::vec3(ellipsoid[0].z, ellipsoid[1].z, ellipsoid[2].z);
			transformInv /= glm::dot(transformInv, transformInv);
			delta.z = glm::dot(delta0, transformInv);
			X[i] += delta;
			V[i] = glm::vec3(0);
		}
	}
}

void StepPhysics(float dt){
	ComputeForces();
	IntegrateImplicit(timeStep);
	EllipsoidCollision();
	ApplyProvotDynamicInverse();
}

void OnRender(GLFWwindow* window, GLuint shaderProgram) {
    startTiming();

    // dynamic vertex positions in the VBO
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, X.size() * sizeof(glm::vec3), &X[0]);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // clear screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(shaderProgram);

    // model-view-projection matrix
    glm::mat4 model = glm::mat4(1.0f);
    glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, dist), glm::vec3(0.0f), Up);
    glm::mat4 projection = glm::perspective(glm::radians(60.0f), (float)width / height, 0.1f, 100.0f);
    glm::mat4 mvp = projection * view * model;

    GLuint mvpLoc = glGetUniformLocation(shaderProgram, "modelViewProjectionMatrix");
    glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(mvp));

    // Bind the VAO
    glBindVertexArray(VAO);

    // Draw the cloth with indexed elements
    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_SHORT, 0);

    // Unbind and clean up
    glBindVertexArray(0);
    glUseProgram(0);
    endTiming();

    // Update window title with FPS and frame time
    updateWindowTitle(fps, frameTime);
}


void OnFramebufferSize(GLFWwindow *window, int width, int height){
	OnReshape(width, height);
}

int main(int argc, char **argv){
	// Initialize GLFW
	if (!glfwInit()){
		fprintf(stderr, "Failed to initialize GLFW\n");
		return -1;
	}

	// Set OpenGL version and profile
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // Required on Mac
	#endif

	// Create window
	window = glfwCreateWindow(width, height, "GLFW Cloth Demo [Implicit Euler Integration]", NULL, NULL);
	if (window == NULL)
	{
		fprintf(stderr, "Failed to create GLFW window\n");
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	// Initialize GLEW after making the context current
	#if defined(__APPLE__)
    // On macOS, GLEW requires this to use modern OpenGL
    	glewExperimental = GL_TRUE;
	#endif
	if (glewInit() != GLEW_OK) {
    fprintf(stderr, "Failed to initialize GLEW\n");
    	return -1;
	}

	// Set callbacks
	glfwSetFramebufferSizeCallback(window, OnFramebufferSize);
	glfwSetMouseButtonCallback(window, OnMouseButton);
	glfwSetCursorPosCallback(window, OnCursorPos);

	// depth test
	glEnable(GL_DEPTH_TEST);

	// Initialize sim
	InitGL();

	GLuint shaderProgram = LoadShaders("vertex_shader.glsl", "fragment_shader.glsl");
	if (shaderProgram == 0) {
        fprintf(stderr, "Failed to load shaders\n");
        return -1;
    }
	// the main loop
	while (!glfwWindowShouldClose(window))
	{
		// frame time
		float newTime = glfwGetTime();
		frameTime = newTime - currentTime;
		currentTime = newTime;
		accumulator += frameTime;

		// Input handling
		glfwPollEvents();

		// Update physics
		if (accumulator >= timeStep)
		{
			StepPhysics(timeStep);
			accumulator -= timeStep;
		}
		OnRender(window, shaderProgram);

		// Swap buffers
		glfwSwapBuffers(window);
	}
	// Cleanup
	OnShutdown();
	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}
