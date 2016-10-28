//
// Lighthouse3D.com OpenGL 3.3 + GLSL 3.3 Sample
//
// Loading and displaying a Textured Model
//
// Uses:
//  Assimp lybrary for model loading
//		http://assimp.sourceforge.net/
//  Devil for image loading
//		http://openil.sourceforge.net/
//	Uniform Blocks
//  Vertex Array Objects
//
// Some parts of the code are strongly based on the Assimp 
// SimpleTextureOpenGL sample that comes with the Assimp 
// distribution, namely the code that relates to loading the images
// and the model.
//
// The code was updated and modified to be compatible with 
// OpenGL 3.3 CORE version
//
// This demo was built for learning purposes only. 
// Some code could be severely optimised, but I tried to 
// keep as simple and clear as possible.
//
// The code comes with no warranties, use it at your own risk.
// You may use it, or parts of it, wherever you want. 
//
// If you do use it I would love to hear about it. Just post a comment
// at Lighthouse3D.com

// Have Fun :-)

#ifdef _WIN32
#pragma comment(lib,"assimp.lib")
#pragma comment(lib,"devil.lib")
#pragma comment(lib,"glew32.lib")

#endif



//OpenCV Includes
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/core/opengl.hpp>


// include DevIL for image loading
#include <IL/il.h>



// include GLEW to access OpenGL 3.3 functions
#include <GL/glew.h>

// GLUT is the toolkit to interface with the OS
#include <GL/freeglut.h>

// auxiliary C file to read the shader text files
#include "textfile.h"

// assimp include files. These three are usually needed.
#include "assimp/Importer.hpp"	//OO version Header!
#include "assimp/postprocess.h"
#include "assimp/scene.h"


#include <math.h>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <iostream>


// Information to render each assimp node
struct MyMesh {

	GLuint vao;
	GLuint texIndex;
	GLuint uniformBlockIndex;
	int numFaces;
};

std::vector<struct MyMesh> myMeshes;

// This is for a shader uniform block
struct MyMaterial {

	float diffuse[4];
	float ambient[4];
	float specular[4];
	float emissive[4];
	float shininess;
	int texCount;
};

//Window Default size
int windowWidth = 512, windowHeight = 512;


// Model Matrix (part of the OpenGL Model View Matrix)
float modelMatrix[16];

// For push and pop matrix
std::vector<float *> matrixStack;

// Vertex Attribute Locations
GLuint vertexLoc = 0, normalLoc = 1, texCoordLoc = 2;

// Uniform Bindings Points
GLuint matricesUniLoc = 1, materialUniLoc = 2;

// The sampler uniform for textured models
// we are assuming a single texture so this will
//always be texture unit 0
GLuint texUnit = 0;

// Uniform Buffer for Matrices
// this buffer will contain 3 matrices: projection, view and model
// each matrix is a float array with 16 components
GLuint matricesUniBuffer;
#define MatricesUniBufferSize sizeof(float) * 16 * 3
#define ProjMatrixOffset 0
#define ViewMatrixOffset sizeof(float) * 16
#define ModelMatrixOffset sizeof(float) * 16 * 2
#define MatrixSize sizeof(float) * 16


// Program and Shader Identifiers
GLuint program, vertexShader, fragmentShader;
GLuint p, vertexShader2D, fragmentShader2D;


// holder for the vertex array object id
GLuint vao, textureID;


// Shader Names
char *vertexFileName = "dirlightdiffambpix.vert";
char *fragmentFileName = "dirlightdiffambpix.frag";

// Create an instance of the Importer class
Assimp::Importer importer;

// the global Assimp scene object
const aiScene* scene = NULL;

// scale factor for the model to fit in the window
float scaleFactor;


// images / texture
// map image filenames to textureIds
// pointer to texture Array
std::map<std::string, GLuint> textureIdMap;

// Replace the model name by your model's filename
static const std::string modelname = "jeep1.ms3d";


// Camera Position
float camX = 0, camY = 0, camZ = 5;

// Mouse Tracking Variables
int startX, startY, tracking = 0;

// Camera Spherical Coordinates
float alpha = 0.0f, beta = 0.0f;
float r = 5.0f;



//our aruco variables
cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_100);
cv::Ptr<cv::aruco::DetectorParameters> detectorParams = cv::aruco::DetectorParameters::create();

std::vector< int > markerIds;
std::vector< std::vector<cv::Point2f> > markerCorners, rejectedCandidates;

cv::VideoCapture cap(0);

double K_[3][3] =
{ { 675, 0, 320 },
{ 0, 675, 240 },
{ 0, 0, 1 } };
cv::Mat K = cv::Mat(3, 3, CV_64F, K_).clone();
const float markerLength = 1.75;
// Distortion coeffs (fill in your actual values here).
double dist_[] = { 0, 0, 0, 0, 0 };
cv::Mat distCoeffs = cv::Mat(5, 1, CV_64F, dist_).clone();
cv::Mat imageMat;

cv::Mat viewMatrix = cv::Mat::zeros(4, 4, CV_32F);



#define M_PI       3.14159265358979323846f


static inline float
DegToRad(float degrees)
{
	return (float)(degrees * (M_PI / 180.0f));
};

// Frame counting and FPS computation
long timet, timebase = 0, frame = 0;
char s[32];

//-----------------------------------------------------------------
// Print for OpenGL errors
//
// Returns 1 if an OpenGL error occurred, 0 otherwise.
//

#define printOpenGLError() printOglError(__FILE__, __LINE__)

int printOglError(char *file, int line)
{

	GLenum glErr;
	int    retCode = 0;

	glErr = glGetError();
	if (glErr != GL_NO_ERROR)
	{
		printf("glError in file %s @ line %d: %s\n",
			file, line, gluErrorString(glErr));
		retCode = 1;
	}
	return retCode;
}


// ----------------------------------------------------
// VECTOR STUFF
//


// res = a cross b;
void crossProduct(float *a, float *b, float *res) {

	res[0] = a[1] * b[2] - b[1] * a[2];
	res[1] = a[2] * b[0] - b[2] * a[0];
	res[2] = a[0] * b[1] - b[0] * a[1];
}


// Normalize a vec3
void normalize(float *a) {

	float mag = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);

	a[0] /= mag;
	a[1] /= mag;
	a[2] /= mag;
}


// ----------------------------------------------------
// MATRIX STUFF
//

// Push and Pop for modelMatrix

void pushMatrix() {

	float *aux = (float *)malloc(sizeof(float) * 16);
	memcpy(aux, modelMatrix, sizeof(float) * 16);
	matrixStack.push_back(aux);
}

void popMatrix() {

	float *m = matrixStack[matrixStack.size() - 1];
	memcpy(modelMatrix, m, sizeof(float) * 16);
	matrixStack.pop_back();
	free(m);
}

// sets the square matrix mat to the identity matrix,
// size refers to the number of rows (or columns)
void setIdentityMatrix(float *mat, int size) {

	// fill matrix with 0s
	for (int i = 0; i < size * size; ++i)
		mat[i] = 0.0f;

	// fill diagonal with 1s
	for (int i = 0; i < size; ++i)
		mat[i + i * size] = 1.0f;
}


//
// a = a * b;
//
void multMatrix(float *a, float *b) {

	float res[16];

	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			res[j * 4 + i] = 0.0f;
			for (int k = 0; k < 4; ++k) {
				res[j * 4 + i] += a[k * 4 + i] * b[j * 4 + k];
			}
		}
	}
	memcpy(a, res, 16 * sizeof(float));

}


// Defines a transformation matrix mat with a translation
void setTranslationMatrix(float *mat, float x, float y, float z) {

	setIdentityMatrix(mat, 4);
	mat[12] = x;
	mat[13] = y;
	mat[14] = z;
}

// Defines a transformation matrix mat with a scale
void setScaleMatrix(float *mat, float sx, float sy, float sz) {

	setIdentityMatrix(mat, 4);
	mat[0] = sx;
	mat[5] = sy;
	mat[10] = sz;
}

// Defines a transformation matrix mat with a rotation 
// angle alpha and a rotation axis (x,y,z)
void setRotationMatrix(float *mat, float angle, float x, float y, float z) {

	float radAngle = DegToRad(angle);
	float co = cos(radAngle);
	float si = sin(radAngle);
	float x2 = x*x;
	float y2 = y*y;
	float z2 = z*z;

	mat[0] = x2 + (y2 + z2) * co;
	mat[4] = x * y * (1 - co) - z * si;
	mat[8] = x * z * (1 - co) + y * si;
	mat[12] = 0.0f;

	mat[1] = x * y * (1 - co) + z * si;
	mat[5] = y2 + (x2 + z2) * co;
	mat[9] = y * z * (1 - co) - x * si;
	mat[13] = 0.0f;

	mat[2] = x * z * (1 - co) - y * si;
	mat[6] = y * z * (1 - co) + x * si;
	mat[10] = z2 + (x2 + y2) * co;
	mat[14] = 0.0f;

	mat[3] = 0.0f;
	mat[7] = 0.0f;
	mat[11] = 0.0f;
	mat[15] = 1.0f;

}

// ----------------------------------------------------
// Model Matrix 
//
// Copies the modelMatrix to the uniform buffer


void setModelMatrix() {

	glBindBuffer(GL_UNIFORM_BUFFER, matricesUniBuffer);
	glBufferSubData(GL_UNIFORM_BUFFER,
		ModelMatrixOffset, MatrixSize, modelMatrix);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

}

// The equivalent to glTranslate applied to the model matrix
void translate(float x, float y, float z) {

	float aux[16];

	setTranslationMatrix(aux, x, y, z);
	multMatrix(modelMatrix, aux);
	setModelMatrix();
}

// The equivalent to glRotate applied to the model matrix
void rotate(float angle, float x, float y, float z) {

	float aux[16];

	setRotationMatrix(aux, angle, x, y, z);
	multMatrix(modelMatrix, aux);
	setModelMatrix();
}

// The equivalent to glScale applied to the model matrix
void scale(float x, float y, float z) {

	float aux[16];

	setScaleMatrix(aux, x, y, z);
	multMatrix(modelMatrix, aux);
	setModelMatrix();
}

// ----------------------------------------------------
// Projection Matrix 
//
// Computes the projection Matrix and stores it in the uniform buffer

void buildProjectionMatrix(float fov, float ratio, float nearp, float farp) {

	float projMatrix[16];

	float f = 1.0f / tan(fov * (M_PI / 360.0f));

	setIdentityMatrix(projMatrix, 4);

	projMatrix[0] = f / ratio;
	projMatrix[1 * 4 + 1] = f;
	projMatrix[2 * 4 + 2] = (farp + nearp) / (nearp - farp);
	projMatrix[3 * 4 + 2] = (2.0f * farp * nearp) / (nearp - farp);
	projMatrix[2 * 4 + 3] = -1.0f;
	projMatrix[3 * 4 + 3] = 0.0f;

	glBindBuffer(GL_UNIFORM_BUFFER, matricesUniBuffer);
	glBufferSubData(GL_UNIFORM_BUFFER, ProjMatrixOffset, MatrixSize, projMatrix);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

}


// ----------------------------------------------------
// View Matrix
//
// Computes the viewMatrix and stores it in the uniform buffer
//
// note: it assumes the camera is not tilted, 
// i.e. a vertical up vector along the Y axis (remember gluLookAt?)
//

void setCamera(float posX, float posY, float posZ,
	float lookAtX, float lookAtY, float lookAtZ) {

	//Set these to make the view matrix happy

	viewMatrix.at<float>(0, 3) = 0;
	viewMatrix.at<float>(1, 3) = 0;
	viewMatrix.at<float>(2, 3) = 0;
	viewMatrix.at<float>(3, 3) = 1;

	glBindBuffer(GL_UNIFORM_BUFFER, matricesUniBuffer);
	glBufferSubData(GL_UNIFORM_BUFFER, ViewMatrixOffset, MatrixSize, (float*)viewMatrix.data);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}




// ----------------------------------------------------------------------------

#define aisgl_min(x,y) (x<y?x:y)
#define aisgl_max(x,y) (y>x?y:x)

void get_bounding_box_for_node(const aiNode* nd,
	aiVector3D* min,
	aiVector3D* max)

{
	aiMatrix4x4 prev;
	unsigned int n = 0, t;

	for (; n < nd->mNumMeshes; ++n) {
		const aiMesh* mesh = scene->mMeshes[nd->mMeshes[n]];
		for (t = 0; t < mesh->mNumVertices; ++t) {

			aiVector3D tmp = mesh->mVertices[t];

			min->x = aisgl_min(min->x, tmp.x);
			min->y = aisgl_min(min->y, tmp.y);
			min->z = aisgl_min(min->z, tmp.z);

			max->x = aisgl_max(max->x, tmp.x);
			max->y = aisgl_max(max->y, tmp.y);
			max->z = aisgl_max(max->z, tmp.z);
		}
	}

	for (n = 0; n < nd->mNumChildren; ++n) {
		get_bounding_box_for_node(nd->mChildren[n], min, max);
	}
}


void get_bounding_box(aiVector3D* min, aiVector3D* max)
{

	min->x = min->y = min->z = 1e10f;
	max->x = max->y = max->z = -1e10f;
	get_bounding_box_for_node(scene->mRootNode, min, max);
}

bool Import3DFromFile(const std::string& pFile)
{

	//check if file exists
	std::ifstream fin(pFile.c_str());
	if (!fin.fail()) {
		fin.close();
	}
	else {
		printf("Couldn't open file: %s\n", pFile.c_str());
		printf("%s\n", importer.GetErrorString());
		return false;
	}

	scene = importer.ReadFile(pFile, aiProcessPreset_TargetRealtime_Quality);

	// If the import failed, report it
	if (!scene)
	{
		printf("%s\n", importer.GetErrorString());
		return false;
	}

	// Now we can access the file's contents.
	printf("Import of scene %s succeeded.", pFile.c_str());

	aiVector3D scene_min, scene_max, scene_center;
	get_bounding_box(&scene_min, &scene_max);
	float tmp;
	tmp = scene_max.x - scene_min.x;
	tmp = scene_max.y - scene_min.y > tmp ? scene_max.y - scene_min.y : tmp;
	tmp = scene_max.z - scene_min.z > tmp ? scene_max.z - scene_min.z : tmp;
	scaleFactor = 1.f / tmp;

	// We're done. Everything will be cleaned up by the importer destructor
	return true;
}


int LoadGLTextures(const aiScene* scene)
{
	ILboolean success;

	/* initialization of DevIL */
	ilInit();

	/* scan scene's materials for textures */
	for (unsigned int m = 0; m<scene->mNumMaterials; ++m)
	{
		int texIndex = 0;
		aiString path;	// filename

		aiReturn texFound = scene->mMaterials[m]->GetTexture(aiTextureType_DIFFUSE, texIndex, &path);
		while (texFound == AI_SUCCESS) {
			//fill map with textures, OpenGL image ids set to 0
			textureIdMap[path.data] = 0;
			// more textures?
			texIndex++;
			texFound = scene->mMaterials[m]->GetTexture(aiTextureType_DIFFUSE, texIndex, &path);
		}
	}

	int numTextures = textureIdMap.size();

	/* create and fill array with DevIL texture ids */
	ILuint* imageIds = new ILuint[numTextures];
	ilGenImages(numTextures, imageIds);

	/* create and fill array with GL texture ids */
	GLuint* textureIds = new GLuint[numTextures];
	glGenTextures(numTextures, textureIds); /* Texture name generation */

											/* get iterator */
	std::map<std::string, GLuint>::iterator itr = textureIdMap.begin();
	int i = 0;
	for (; itr != textureIdMap.end(); ++i, ++itr)
	{
		//save IL image ID
		std::string filename = (*itr).first;  // get filename
		(*itr).second = textureIds[i];	  // save texture id for filename in map

		ilBindImage(imageIds[i]); /* Binding of DevIL image name */
		ilEnable(IL_ORIGIN_SET);
		ilOriginFunc(IL_ORIGIN_LOWER_LEFT);
		success = ilLoadImage((ILstring)filename.c_str());

		if (success) {
			/* Convert image to RGBA */
			ilConvertImage(IL_RGBA, IL_UNSIGNED_BYTE);

			/* Create and load textures to OpenGL */
			glBindTexture(GL_TEXTURE_2D, textureIds[i]);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ilGetInteger(IL_IMAGE_WIDTH),
				ilGetInteger(IL_IMAGE_HEIGHT), 0, GL_RGBA, GL_UNSIGNED_BYTE,
				ilGetData());
		}
		else
			printf("Couldn't load Image: %s\n", filename.c_str());
	}
	/* Because we have already copied image data into texture data
	we can release memory used by image. */
	ilDeleteImages(numTextures, imageIds);

	//Cleanup
	delete[] imageIds;
	delete[] textureIds;

	//return success;
	return true;
}



//// Can't send color down as a pointer to aiColor4D because AI colors are ABGR.
//void Color4f(const aiColor4D *color)
//{
//	glColor4f(color->r, color->g, color->b, color->a);
//}

void set_float4(float f[4], float a, float b, float c, float d)
{
	f[0] = a;
	f[1] = b;
	f[2] = c;
	f[3] = d;
}

void color4_to_float4(const aiColor4D *c, float f[4])
{
	f[0] = c->r;
	f[1] = c->g;
	f[2] = c->b;
	f[3] = c->a;
}



void genVAOsAndUniformBuffer(const aiScene *sc) {

	struct MyMesh aMesh;
	struct MyMaterial aMat;
	GLuint buffer;

	// For each mesh
	for (unsigned int n = 0; n < sc->mNumMeshes; ++n)
	{
		const aiMesh* mesh = sc->mMeshes[n];

		// create array with faces
		// have to convert from Assimp format to array
		unsigned int *faceArray;
		faceArray = (unsigned int *)malloc(sizeof(unsigned int) * mesh->mNumFaces * 3);
		unsigned int faceIndex = 0;

		for (unsigned int t = 0; t < mesh->mNumFaces; ++t) {
			const aiFace* face = &mesh->mFaces[t];

			memcpy(&faceArray[faceIndex], face->mIndices, 3 * sizeof(unsigned int));
			faceIndex += 3;
		}
		aMesh.numFaces = sc->mMeshes[n]->mNumFaces;

		// generate Vertex Array for mesh
		glGenVertexArrays(1, &(aMesh.vao));
		glBindVertexArray(aMesh.vao);

		// buffer for faces
		glGenBuffers(1, &buffer);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * mesh->mNumFaces * 3, faceArray, GL_STATIC_DRAW);

		// buffer for vertex positions
		if (mesh->HasPositions()) {
			glGenBuffers(1, &buffer);
			glBindBuffer(GL_ARRAY_BUFFER, buffer);
			glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * mesh->mNumVertices, mesh->mVertices, GL_STATIC_DRAW);
			glEnableVertexAttribArray(vertexLoc);
			glVertexAttribPointer(vertexLoc, 3, GL_FLOAT, 0, 0, 0);
		}

		// buffer for vertex normals
		if (mesh->HasNormals()) {
			glGenBuffers(1, &buffer);
			glBindBuffer(GL_ARRAY_BUFFER, buffer);
			glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * mesh->mNumVertices, mesh->mNormals, GL_STATIC_DRAW);
			glEnableVertexAttribArray(normalLoc);
			glVertexAttribPointer(normalLoc, 3, GL_FLOAT, 0, 0, 0);
		}

		// buffer for vertex texture coordinates
		if (mesh->HasTextureCoords(0)) {
			float *texCoords = (float *)malloc(sizeof(float) * 2 * mesh->mNumVertices);
			for (unsigned int k = 0; k < mesh->mNumVertices; ++k) {

				texCoords[k * 2] = mesh->mTextureCoords[0][k].x;
				texCoords[k * 2 + 1] = mesh->mTextureCoords[0][k].y;

			}
			glGenBuffers(1, &buffer);
			glBindBuffer(GL_ARRAY_BUFFER, buffer);
			glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 2 * mesh->mNumVertices, texCoords, GL_STATIC_DRAW);
			glEnableVertexAttribArray(texCoordLoc);
			glVertexAttribPointer(texCoordLoc, 2, GL_FLOAT, 0, 0, 0);
		}

		// unbind buffers
		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		// create material uniform buffer
		aiMaterial *mtl = sc->mMaterials[mesh->mMaterialIndex];

		aiString texPath;	//contains filename of texture
		if (AI_SUCCESS == mtl->GetTexture(aiTextureType_DIFFUSE, 0, &texPath)) {
			//bind texture
			unsigned int texId = textureIdMap[texPath.data];
			aMesh.texIndex = texId;
			aMat.texCount = 1;
		}
		else
			aMat.texCount = 0;

		float c[4];
		set_float4(c, 0.8f, 0.8f, 0.8f, 1.0f);
		aiColor4D diffuse;
		if (AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_DIFFUSE, &diffuse))
			color4_to_float4(&diffuse, c);
		memcpy(aMat.diffuse, c, sizeof(c));

		set_float4(c, 0.2f, 0.2f, 0.2f, 1.0f);
		aiColor4D ambient;
		if (AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_AMBIENT, &ambient))
			color4_to_float4(&ambient, c);
		memcpy(aMat.ambient, c, sizeof(c));

		set_float4(c, 0.0f, 0.0f, 0.0f, 1.0f);
		aiColor4D specular;
		if (AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_SPECULAR, &specular))
			color4_to_float4(&specular, c);
		memcpy(aMat.specular, c, sizeof(c));

		set_float4(c, 0.0f, 0.0f, 0.0f, 1.0f);
		aiColor4D emission;
		if (AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_EMISSIVE, &emission))
			color4_to_float4(&emission, c);
		memcpy(aMat.emissive, c, sizeof(c));

		float shininess = 0.0;
		unsigned int max;
		aiGetMaterialFloatArray(mtl, AI_MATKEY_SHININESS, &shininess, &max);
		aMat.shininess = shininess;

		glGenBuffers(1, &(aMesh.uniformBlockIndex));
		glBindBuffer(GL_UNIFORM_BUFFER, aMesh.uniformBlockIndex);
		glBufferData(GL_UNIFORM_BUFFER, sizeof(aMat), (void *)(&aMat), GL_STATIC_DRAW);

		myMeshes.push_back(aMesh);
	}
}


// ------------------------------------------------------------
//
// Reshape Callback Function
//

void changeSize(int w, int h) {

	float ratio;
	// Prevent a divide by zero, when window is too short
	// (you cant make a window of zero width).
	if (h == 0)
		h = 1;

	windowWidth = w;
	windowHeight = h;

	// Set the viewport to be the entire window
	glViewport(0, 0, w, h);

	ratio = (1.0f * w) / h;
	buildProjectionMatrix(53.13f, ratio, 0.1f, 10.0f);
}


// ------------------------------------------------------------
//
// Render stuff
//

// Render Assimp Model

void recursive_render(const aiScene *sc, const aiNode* nd)
{

	// Get node transformation matrix
	aiMatrix4x4 m = nd->mTransformation;
	// OpenGL matrices are column major
	m.Transpose();

	// save model matrix and apply node transformation
	pushMatrix();

	float aux[16];
	memcpy(aux, &m, sizeof(float) * 16);
	multMatrix(modelMatrix, aux);
	setModelMatrix();


	// draw all meshes assigned to this node
	for (unsigned int n = 0; n < nd->mNumMeshes; ++n) {
		// bind material uniform
		glBindBufferRange(GL_UNIFORM_BUFFER, materialUniLoc, myMeshes[nd->mMeshes[n]].uniformBlockIndex, 0, sizeof(struct MyMaterial));
		// bind texture
		glBindTexture(GL_TEXTURE_2D, myMeshes[nd->mMeshes[n]].texIndex);
		// bind VAO
		glBindVertexArray(myMeshes[nd->mMeshes[n]].vao);
		// draw
		glDrawElements(GL_TRIANGLES, myMeshes[nd->mMeshes[n]].numFaces * 3, GL_UNSIGNED_INT, 0);

	}

	// draw all children
	for (unsigned int n = 0; n < nd->mNumChildren; ++n) {
		recursive_render(sc, nd->mChildren[n]);
	}
	popMatrix();
}


// Rendering Callback Function

void detectArucoMarkers() {
	cv::aruco::detectMarkers(
		imageMat,		// input image
		dictionary,		// type of markers that will be searched for
		markerCorners,	// output vector of marker corners
		markerIds,		// detected marker IDs
		detectorParams,	// algorithm parameters
		rejectedCandidates);

	if (markerIds.size() > 0) {
		// Draw all detected markers.
		cv::aruco::drawDetectedMarkers(imageMat, markerCorners, markerIds);

		std::vector< cv::Vec3d > rvecs, tvecs;
		cv::aruco::estimatePoseSingleMarkers(
			markerCorners,	// vector of already detected markers corners
			markerLength,	// length of the marker's side
			K,				// input 3x3 floating-point instrinsic camera matrix K
			distCoeffs,		// vector of distortion coefficients of 4, 5, 8 or 12 elements
			rvecs,			// array of output rotation vectors 
			tvecs);			// array of output translation vectors

		for (unsigned int i = 0; i < markerIds.size(); i++) {
			cv::Vec3d r = rvecs[i];
			cv::Vec3d t = tvecs[i];
			if (markerIds[i] == 0) {
				cv::Mat rot;
				Rodrigues(rvecs[i], rot);
				for (unsigned int row = 0; row < 3; ++row)
				{
					for (unsigned int col = 0; col < 3; ++col)
					{
						viewMatrix.at<float>(row, col) = (float)rot.at<double>(row, col);
					}
					viewMatrix.at<float>(row, 3) = (float)tvecs[i][row] * 0.1f;
				}
				viewMatrix.at<float>(3, 3) = 1.0f;

				cv::Mat cvToGl = cv::Mat::zeros(4, 4, CV_32F);
				cvToGl.at<float>(0, 0) = 1.0f;
				cvToGl.at<float>(1, 1) = -1.0f; // Invert the y axis 
				cvToGl.at<float>(2, 2) = -1.0f; // invert the z axis 
				cvToGl.at<float>(3, 3) = 1.0f;
				viewMatrix = cvToGl * viewMatrix;
				cv::transpose(viewMatrix, viewMatrix);

			}

			// Draw coordinate axes.
			cv::aruco::drawAxis(imageMat,
				K, distCoeffs,			// camera parameters
				r, t,					// marker pose
				0.5*markerLength);		// length of the axes to be drawn

										// Draw a symbol in the upper right corner of the detected marker.
		}
	}
}



void camTimer(int value) {
	// Capture next frame
	cap >> imageMat; // get image from camera
					 //Detect the aruco markers and draw their orientation
	detectArucoMarkers();

	IplImage *image;
	IplImage copy = imageMat;
	image = &copy;


	// Convert to RGB
	cvCvtColor(image, image, CV_BGR2RGB);

	// Create Texture
	//gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGB, image->width, image->height, GL_RGB, GL_UNSIGNED_BYTE, image->imageData);

	glutTimerFunc(1000.0f / 15.0f, camTimer, 0);
}

// ------------------------------------------------------------
//
//			Prepare texture
//
// ------------------------------------------------------------

void prepareTexture(int w, int h, unsigned char* data) {

	/* Create and load texture to OpenGL */
	glGenTextures(1, &textureID); /* Texture name generation */
	glBindTexture(GL_TEXTURE_2D, textureID); 
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 
                w, h, 
                0, GL_RGBA, GL_UNSIGNED_BYTE,
                data); 
	glGenerateMipmap(GL_TEXTURE_2D);
}


void renderScene(void) {

// clear the framebuffer (color and depth)
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	// Capture next frame
	cap >> imageMat; // get image from camera


	IplImage *image;
	IplImage copy = imageMat;
	image = &copy;


	// Convert to RGB
	cvCvtColor(image, image, CV_BGR2RGB);

	// Create Texture
	glGenTextures(1, &textureID); /* Texture name generation */
	glBindTexture(GL_TEXTURE_2D, textureID); 
	gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGB, image->width, image->height, GL_RGB, GL_UNSIGNED_BYTE, image->imageData);
	

	// Use the program p
	glUseProgram(p);
	// Bind the vertex array object
	glBindVertexArray(vao);
	// Bind texture
	glBindTexture(GL_TEXTURE_2D, textureID);
	// draw the 6 vertices
	glDrawArrays(GL_TRIANGLES, 0, 6);



	glClear( GL_DEPTH_BUFFER_BIT);
		// set camera matrix
	setCamera(camX, camY, camZ, 0, 0, 0);

	// set the model matrix to the identity Matrix
	setIdentityMatrix(modelMatrix, 4);

	// sets the model matrix to a scale matrix so that the model fits in the window
	scale(scaleFactor, scaleFactor, scaleFactor);

	// keep rotating the model
	//rotate(step, 0.0f, 1.0f, 0.0f);

	// use our shader
	glUseProgram(program);

	// we are only going to use texture unit 0
	// unfortunately samplers can't reside in uniform blocks
	// so we have set this uniform separately
	glUniform1i(texUnit, 0);
		

	//glLoadMatrixf((float*)viewMatrix.data);
	recursive_render(scene, scene->mRootNode);



	// swap buffers
	glutSwapBuffers();
}


// ------------------------------------------------------------
//
// Events from the Keyboard
//

void processKeys(unsigned char key, int xx, int yy)
{
	switch (key) {

	case 27:

		glutLeaveMainLoop();
		break;

	case 'z': r -= 0.1f; break;
	case 'x': r += 0.1f; break;
	case 'm': glEnable(GL_MULTISAMPLE); break;
	case 'n': glDisable(GL_MULTISAMPLE); break;
	}
	camX = r * sin(alpha * 3.14f / 180.0f) * cos(beta * 3.14f / 180.0f);
	camZ = r * cos(alpha * 3.14f / 180.0f) * cos(beta * 3.14f / 180.0f);
	camY = r *   						     sin(beta * 3.14f / 180.0f);

	//  uncomment this if not using an idle func
	//	glutPostRedisplay();
}


// ------------------------------------------------------------
//
// Mouse Events
//

void processMouseButtons(int button, int state, int xx, int yy)
{
	// start tracking the mouse
	if (state == GLUT_DOWN) {
		startX = xx;
		startY = yy;
		if (button == GLUT_LEFT_BUTTON)
			tracking = 1;
		else if (button == GLUT_RIGHT_BUTTON)
			tracking = 2;
	}

	//stop tracking the mouse
	else if (state == GLUT_UP) {
		if (tracking == 1) {
			alpha += (startX - xx);
			beta += (yy - startY);
		}
		else if (tracking == 2) {
			r += (yy - startY) * 0.01f;
		}
		tracking = 0;
	}
}

// Track mouse motion while buttons are pressed

void processMouseMotion(int xx, int yy)
{

	int deltaX, deltaY;
	float alphaAux, betaAux;
	float rAux;

	deltaX = startX - xx;
	deltaY = yy - startY;

	// left mouse button: move camera
	if (tracking == 1) {


		alphaAux = alpha + deltaX;
		betaAux = beta + deltaY;

		if (betaAux > 85.0f)
			betaAux = 85.0f;
		else if (betaAux < -85.0f)
			betaAux = -85.0f;

		rAux = r;

		camX = rAux * cos(betaAux * 3.14f / 180.0f) * sin(alphaAux * 3.14f / 180.0f);
		camZ = rAux * cos(betaAux * 3.14f / 180.0f) * cos(alphaAux * 3.14f / 180.0f);
		camY = rAux * sin(betaAux * 3.14f / 180.0f);
	}
	// right mouse button: zoom
	else if (tracking == 2) {

		alphaAux = alpha;
		betaAux = beta;
		rAux = r + (deltaY * 0.01f);

		camX = rAux * cos(betaAux * 3.14f / 180.0f) * sin(alphaAux * 3.14f / 180.0f);
		camZ = rAux * cos(betaAux * 3.14f / 180.0f) * cos(alphaAux * 3.14f / 180.0f);
		camY = rAux * sin(betaAux * 3.14f / 180.0f);
	}


	//  uncomment this if not using an idle func
	//	glutPostRedisplay();
}




void mouseWheel(int wheel, int direction, int x, int y) {

	r += direction * 0.1f;
	camX = r * sin(alpha * 3.14f / 180.0f) * cos(beta * 3.14f / 180.0f);
	camZ = r * cos(alpha * 3.14f / 180.0f) * cos(beta * 3.14f / 180.0f);
	camY = r *   						     sin(beta * 3.14f / 180.0f);
}






// --------------------------------------------------------
//
// Shader Stuff
//

void printShaderInfoLog(GLuint obj)
{
	int infologLength = 0;
	int charsWritten = 0;
	char *infoLog;

	glGetShaderiv(obj, GL_INFO_LOG_LENGTH, &infologLength);

	if (infologLength > 0)
	{
		infoLog = (char *)malloc(infologLength);
		glGetShaderInfoLog(obj, infologLength, &charsWritten, infoLog);
		printf("%s\n", infoLog);
		free(infoLog);
	}
}


void printProgramInfoLog(GLuint obj)
{
	int infologLength = 0;
	int charsWritten = 0;
	char *infoLog;

	glGetProgramiv(obj, GL_INFO_LOG_LENGTH, &infologLength);

	if (infologLength > 0)
	{
		infoLog = (char *)malloc(infologLength);
		glGetProgramInfoLog(obj, infologLength, &charsWritten, infoLog);
		printf("%s\n", infoLog);
		free(infoLog);
	}
}


GLuint setupShaders() {

	char *vs = NULL, *fs = NULL, *fs2 = NULL;

	GLuint p, v, f;

	v = glCreateShader(GL_VERTEX_SHADER);
	f = glCreateShader(GL_FRAGMENT_SHADER);

	vs = textFileRead(vertexFileName);
	fs = textFileRead(fragmentFileName);

	const char * vv = vs;
	const char * ff = fs;

	glShaderSource(v, 1, &vv, NULL);
	glShaderSource(f, 1, &ff, NULL);

	free(vs); free(fs);

	glCompileShader(v);
	glCompileShader(f);

	p = glCreateProgram();
	glAttachShader(p, v);
	glAttachShader(p, f);

	glBindFragDataLocation(p, 0, "output");

	glBindAttribLocation(p, vertexLoc, "position");
	glBindAttribLocation(p, normalLoc, "normal");
	glBindAttribLocation(p, texCoordLoc, "texCoord");

	glLinkProgram(p);
	glValidateProgram(p);

	program = p;
	vertexShader = v;
	fragmentShader = f;

	GLuint k = glGetUniformBlockIndex(p, "Matrices");
	glUniformBlockBinding(p, k, matricesUniLoc);
	glUniformBlockBinding(p, glGetUniformBlockIndex(p, "Material"), materialUniLoc);

	texUnit = glGetUniformLocation(p, "texUnit");

	return(p);
}

// --------------------------------------------------------
//
//			Shader Stuff
//
// --------------------------------------------------------

void setupShaders2D() {

	// variables to hold the shader's source code
	char *vs = NULL, *fs = NULL;

	// holders for the shader's ids
	GLuint v, f;

	// create the two shaders
	v = glCreateShader(GL_VERTEX_SHADER);
	f = glCreateShader(GL_FRAGMENT_SHADER);

	// read the source code from file
	vs = textFileRead("texture.vert");
	fs = textFileRead("texture.frag");

	// castings for calling the shader source function
	const char * vv = vs;
	const char * ff = fs;

	// setting the source for each shader
	glShaderSource(v, 1, &vv, NULL);
	glShaderSource(f, 1, &ff, NULL);

	// free the source strings
	free(vs); free(fs);

	// compile the sources
	glCompileShader(v);
	glCompileShader(f);

	// create a program and attach the shaders
	p = glCreateProgram();
	glAttachShader(p, v);
	glAttachShader(p, f);

	// Bind the fragment data output variable location
	// requires linking afterwards
	glBindFragDataLocation(p, 0, "outputF");

	// link the program
	glLinkProgram(p);

	GLint myLoc = glGetUniformLocation(p, "texUnit");
	//glProgramUniform1d(p, myLoc, 0);
}



int init2D() {
	// Data for the two triangles
	float position[] = { -1.0f, -1.0f, 0.0f, 1.0f,
		1.0f,  1.0f, 0.0f, 1.0f,
		-1.0f,  1.0f, 0.0f, 1.0f,

		1.0f,  1.0f, 0.0f, 1.0f,
		-1.0f, -1.0f, 0.0f, 1.0f,
		1.0f, -1.0f, 0.5f, 1.0f,
	};

	float textureCoord[] = {
		0.0f, 0.0f,
		1.0f, 1.0f,
		0.0f, 1.0f,

		1.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f };


	// variables to hold the shader's source code
	char *vs = NULL, *fs = NULL;

	// holders for the shader's ids
	GLuint v, f;

	// create the two shaders
	v = glCreateShader(GL_VERTEX_SHADER);
	f = glCreateShader(GL_FRAGMENT_SHADER);

	// read the source code from file
	vs = textFileRead("texture.vert");
	fs = textFileRead("texture.frag");

	// castings for calling the shader source function
	const char * vv = vs;
	const char * ff = fs;

	// setting the source for each shader
	glShaderSource(v, 1, &vv, NULL);
	glShaderSource(f, 1, &ff, NULL);

	// free the source strings
	free(vs); free(fs);

	// compile the sources
	glCompileShader(v);
	glCompileShader(f);

	// create a program and attach the shaders
	p = glCreateProgram();
	glAttachShader(p, v);
	glAttachShader(p, f);

	// Bind the fragment data output variable location
	// requires linking afterwards
	glBindFragDataLocation(p, 0, "outputF");

	// link the program
	glLinkProgram(p);

	GLint myLoc = glGetUniformLocation(p, "texUnit");
	//glProgramUniform1d(p, myLoc, 0);

	GLuint vertexLoc, texCoordLoc;

	// Get the locations of the attributes in the current program
	vertexLoc = glGetAttribLocation(p, "position");
	texCoordLoc = glGetAttribLocation(p, "texCoord");

	// Generate and bind a Vertex Array Object
	// this encapsulates the buffers used for drawing the triangle
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

	// Generate two slots for the position and color buffers
    GLuint buffers[2];
    glGenBuffers(2, buffers);

    // bind buffer for vertices and copy data into buffer
    glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(position), position, GL_STATIC_DRAW);
    glEnableVertexAttribArray(vertexLoc);
    glVertexAttribPointer(vertexLoc, 4, GL_FLOAT, 0, 0, 0);
 
    // bind buffer for normals and copy data into buffer
    glBindBuffer(GL_ARRAY_BUFFER, buffers[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(textureCoord), textureCoord, GL_STATIC_DRAW);
    glEnableVertexAttribArray(texCoordLoc);
    glVertexAttribPointer(texCoordLoc, 2, GL_FLOAT, 0, 0, 0);


	return true;
}



// ------------------------------------------------------------
//
// Model loading and OpenGL setup
//


int init()
{
	if (!Import3DFromFile(modelname))
		return(0);

	LoadGLTextures(scene);

	glGetUniformBlockIndex = (PFNGLGETUNIFORMBLOCKINDEXPROC)glutGetProcAddress("glGetUniformBlockIndex");
	glUniformBlockBinding = (PFNGLUNIFORMBLOCKBINDINGPROC)glutGetProcAddress("glUniformBlockBinding");
	glGenVertexArrays = (PFNGLGENVERTEXARRAYSPROC)glutGetProcAddress("glGenVertexArrays");
	glBindVertexArray = (PFNGLBINDVERTEXARRAYPROC)glutGetProcAddress("glBindVertexArray");
	glBindBufferRange = (PFNGLBINDBUFFERRANGEPROC)glutGetProcAddress("glBindBufferRange");
	glDeleteVertexArrays = (PFNGLDELETEVERTEXARRAYSPROC)glutGetProcAddress("glDeleteVertexArrays");

	
	
	genVAOsAndUniformBuffer(scene);

	glEnable(GL_DEPTH_TEST);
	glClearColor(0, 0, 0, 0.0f);

	//
	// Uniform Block
	//
	glGenBuffers(1, &matricesUniBuffer);
	glBindBuffer(GL_UNIFORM_BUFFER, matricesUniBuffer);
	glBufferData(GL_UNIFORM_BUFFER, MatricesUniBufferSize, NULL, GL_DYNAMIC_DRAW);
	glBindBufferRange(GL_UNIFORM_BUFFER, matricesUniLoc, matricesUniBuffer, 0, MatricesUniBufferSize);	//setUniforms();
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	glEnable(GL_MULTISAMPLE);

	init2D();



	return true;
}


void myTimer(int value) {
	glutPostRedisplay();
	glutTimerFunc(1000.0f / 60.0f, myTimer, 0);
	
}







// ------------------------------------------------------------
//
// Main function
//


int main(int argc, char **argv) {

	//  GLUT initialization
	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA | GLUT_MULTISAMPLE);

	glutInitContextVersion(3, 3);
	glutInitContextFlags(GLUT_COMPATIBILITY_PROFILE);


	glutInitWindowPosition(100, 100);
	glutInitWindowSize(windowWidth, windowHeight);
	glutCreateWindow("Lighthouse3D - Assimp Demo");


	//  Callback Registration
	glutDisplayFunc(renderScene);
	glutReshapeFunc(changeSize);
	glutTimerFunc(1000.0f / 60.0f, myTimer, 0);
	glutTimerFunc(1000.0f / 15.0f, camTimer, 0);

	//	Mouse and Keyboard Callbacks
	glutKeyboardFunc(processKeys);
	glutMouseFunc(processMouseButtons);
	glutMotionFunc(processMouseMotion);

	glutMouseWheelFunc(mouseWheel);

	//	Init GLEW
	//glewExperimental = GL_TRUE;
	glewInit();
	if (glewIsSupported("GL_VERSION_3_3"))
		printf("Ready for OpenGL 3.3\n");
	else {
		printf("OpenGL 3.3 not supported\n");
		return(1);
	}

	//  Init the app (load model and textures) and OpenGL
	if (!init())
		printf("Could not Load the Model\n");

	printf("Vendor: %s\n", glGetString(GL_VENDOR));
	printf("Renderer: %s\n", glGetString(GL_RENDERER));
	printf("Version: %s\n", glGetString(GL_VERSION));
	printf("GLSL: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));


	// return from main loop
	//glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);


	program = setupShaders();
	//setupShaders2D();
	

	//  GLUT main loop
	glutMainLoop();

	// cleaning up
	textureIdMap.clear();

	// clear myMeshes stuff
	for (unsigned int i = 0; i < myMeshes.size(); ++i) {

		glDeleteVertexArrays(1, &(myMeshes[i].vao));
		glDeleteTextures(1, &(myMeshes[i].texIndex));
		glDeleteBuffers(1, &(myMeshes[i].uniformBlockIndex));
	}
	// delete buffers
	glDeleteBuffers(1, &matricesUniBuffer);

	return(0);
}

