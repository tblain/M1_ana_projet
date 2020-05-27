#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core/mat.hpp>

#include <queue>
#include <random>
#include <typeinfo>
#include <chrono>

#include <stdio.h>
#include <iostream>

#define MAXFRAMESKIP 10

using namespace cv;
using namespace std;
using namespace std::chrono;

int frameSkip = 0;

typedef cv::Point3_<uint8_t> Pixel;

uint64_t timeSinceEpochMillisec() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

// =========== Segmentor =======================================================

class Segmentor {
    public:
	// =========== Variables ===============================================

	Mat img;

	// nb de germes
	int nb_germes;

	int lar;
	int hau;

	// nb de pixels dans un groupe
	int nb_in_groupes;

	vector<int> *groupes;

	// pour chaque groupe: le nb de pixels a l'interieur
	int *nb_in_groupe;

	// vrai groupe de chaque groupe;
	int *val_groupe;

	// tableau contenant pour chaque point l'index du groupe auquel il appartient ou alors 0
	Mat tab;

	// tableau contenant pour chaque groupe la couleur moyenne
	double *avg_grp_col;
	double *sum_grp_col;

	std::queue<int> queue;

	int step;
	int nb_pixels;

	double merge_thresh;
	double grow_thresh;

	bool display_true_col;

	// =========== Methodes ================================================

	// ==========================================================================
	int* create_germes_regular(void) {
	    int* germes = new int[2 * (this->nb_germes+1)];
	    int dist_between_pts = this->nb_pixels / this->nb_germes;

	    for (int i = 2; i < (this->nb_germes+1)*2; i+=2) {
		int n = dist_between_pts * i / 2;
		int x = n % this->lar;
		int y = n / this->lar;

		germes[i] = x;
		germes[i+1] = y;
	    }

	    return germes;
	}

	// ==========================================================================

	int* create_germes_square(void) {
	    uint64_t t1 = timeSinceEpochMillisec();
	    // germe vector

	    // square size
	    int ss = 25;

	    int shau = this->hau / ss;
	    int slar = this->hau / ss;

	    cv::Mat kernelH(3, 3, CV_32F);
	    cv::Mat kernelV(3, 3, CV_32F);

	    for (int i = 0; i < 3; ++i) {
		kernelH.at<float>(i,0) = 1.0f;
		kernelH.at<float>(i,1) = 0.0f;
		kernelH.at<float>(i,2) = -1.0f;

		kernelV.at<float>(0,i) = 1.0f;
		kernelV.at<float>(1,i) = 0.0f;
		kernelV.at<float>(2,i) = -1.0f;
	    }

	    Mat edge1, edge2, res;

	    cv::filter2D(this->img, edge1, -1, kernelH);
	    cv::filter2D(this->img, edge2, -1, kernelV);

	    abs(edge1);
	    abs(edge2);

	    cv::add(edge1, edge2, res);

	    int* changes = new int[shau * slar];

	    // mis a zero de changes
	    for (int i = 0; i < slar; ++i) {
		for (int j = 0; j < shau; ++j) {
		    changes[i * slar + j] = 0;
		}
	    }

	    this->img.forEach<Pixel> ( [changes, slar, shau, res, ss](Pixel &pixel, const int * position) -> void {
		changes[ (position[0] / ss ) * slar + (position[1] / ss) ] += (int) res.at<uchar>(position[0], position[1]);
	      }
	    );

	    std::default_random_engine generator;
	    std::uniform_int_distribution<int> distribution_coord(0, ss-1);
	    std::uniform_int_distribution<int> distribution_spawn(0, 20);

	    int x, y;
	    this->nb_germes = 1;

	    int size = shau * slar * 8 * 2;
	    int* germes = new int[size];

	    int circled = 2;
	    int pal1=0, pal1fond=0, pal2=0, pal3=0;
	    int pal1_thres = 100, pal2thres = 150;
	    int change;

	    for (int i = 0; i < slar; ++i) {
		for (int j = 0; j < shau; ++j) {
		    change = changes[i * slar + j];

		    //cout << "i: " << i << " / j: " << j << " / change: " << change <<endl;

		    if(change < pal1_thres) {
			pal1++;

			circled = 2;

			if(i + 1 < slar && changes[(i+1) * slar + j] > pal1_thres + 50)
			    circled--;
			if(i - 1 >= 0 && changes[(i-1) * slar + j] > pal1_thres + 50)
			    circled--;
			if(j - 1 < shau && changes[i * slar + (j-1)] > pal1_thres + 50)
			    circled--;
			if(j - 1 >= 0 && changes[i * slar + (j-1)] > pal1_thres + 50)
			    circled--;

			if(circled >= 0 && distribution_spawn(generator) == 0) {
			    pal1fond++;

			    x = i * ss + distribution_coord(generator);
			    y = j * ss + distribution_coord(generator);
			    germes[this->nb_germes*2] = x;
			    germes[this->nb_germes*2+1] = y;
			    this->nb_germes++;

			} else {

			    if(distribution_spawn(generator) < 20 - 5) {
				x = i * ss + distribution_coord(generator);
				y = j * ss + distribution_coord(generator);
				germes[this->nb_germes*2] = x;
				germes[this->nb_germes*2+1] = y;
				this->nb_germes++;
			    }
			}

		    } else if(change < 100) {
			pal2++;

			for (int q = 0; q < 2; ++q) {
			    x = i + distribution_coord(generator);
			    y = j + distribution_coord(generator);
			    germes[this->nb_germes*2] = x;
			    germes[this->nb_germes*2+1] = y;
			    this->nb_germes++;
			}

		    } else {
			pal3++;

			for (int q = 0; q < 1; ++q) {
			    x = i + distribution_coord(generator);
			    y = j + distribution_coord(generator);
			    germes[this->nb_germes*2] = x;
			    germes[this->nb_germes*2+1] = y;
			    this->nb_germes++;
			}

		    }
		}
	    }

	    cout << "changes: " << shau * slar << endl;
	    cout << "changes dim: " << shau << " / " << slar << endl;
	    cout << "tailes germes: " << size << endl;
	    cout << "pal1: " << pal1 << endl;
	    cout << "pal1fond: " << pal1fond << endl;
	    cout << "pal2: " << pal2 << endl;
	    cout << "pal3: " << pal3 << endl;
	    cout << "nb germes: " << this->nb_germes << endl;
	    cout << "nb germes * 2: " << (this->nb_germes+1) * 2 - 2 << endl;

	    uint64_t t2 = timeSinceEpochMillisec();
	    std::cout << "\nTemps Germes square: " << t2 - t1 << " ms" << std::endl;

	    return germes;
	}

	// ==========================================================================
	int* create_germes_random(void) {
	    int* germes = new int[2 * (this->nb_germes+1)];

	    std::default_random_engine generator;
	    std::uniform_int_distribution<int> distributionx(0,this->lar);
	    std::uniform_int_distribution<int> distributiony(0,this->hau);

	    for (int i = 2; i < (this->nb_germes+1)*2; i+=2) {
		germes[i] = distributionx(generator);
		germes[i+1] = distributiony(generator);
	    }

	    return germes;
	}

	// ==========================================================================
	void init_segmentation(void) {
	    int* germes = this->create_germes_regular();

	    Size img_size = img.size();
	    cout << "size: " << img_size << endl;
	    this->tab = cv::Mat::zeros(img_size, CV_32S);
	    this->tab.create(img_size, CV_32S);
	    this->avg_grp_col = new double[this->nb_germes+1];
	    this->sum_grp_col = new double[this->nb_germes+1];
	    this->nb_in_groupe = new int[this->nb_germes+1];
	    this->nb_in_groupes = 0;

	    this->groupes = new std::vector<int>[this->nb_germes+1];
	    this->val_groupe = new int[this->nb_germes+1];

	    this->val_groupe[0] = 0;
	    for (int i = 1; i < nb_germes+1; ++i) {
		this->val_groupe[i] = i;
	    }

	    int x, y, g;
	    // on instancie les seeds
	    for (int i = 2; i < (this->nb_germes+1)*2 - 2; i+=2) {
		x = germes[i];
		y = germes[i+1];
		//cout << "x: " << x << endl;

		g = i/2;

		this->nb_in_groupes++;
		this->tab.at<int>(x, y) = g;
		this->avg_grp_col[g] = this->img.at<uchar>(x, y);
		this->sum_grp_col[g] = this->img.at<uchar>(x, y);
		this->nb_in_groupe[g] = 1;

		this->queue.push(x);
		this->queue.push(y);
		this->queue.push(g);
	    }

	    delete[] germes;
	}
	// ==========================================================================
	void pre_seg_dist_calcul(void) {
	    //printf("pre_seg_dist_calcul\n");

	}

	// ==========================================================================
	Mat segmentation(Mat img, int nb_germes, double merge_thresh, double grow_thresh, bool display_true_col) {
	    this->img = img;
	    this->nb_germes = nb_germes;

	    this->lar = img.size[0];
	    this->hau = img.size[1];
	    this->nb_pixels = this->hau * this->lar;
	    this->display_true_col = display_true_col ;

	    this->merge_thresh = merge_thresh;
	    this->grow_thresh = grow_thresh;

	    this->init_segmentation();
	    this->pre_seg_dist_calcul();

	    this->step = 0;

	    this->grow();

	    this->merge();

	    Mat img_seg = this->create_seg_img();

	    this->trace_contour(img_seg);

	    return img_seg;
	}

	// ==========================================================================
	void grow(void) {
	    std::cout << "=================================" << std::endl;
	    printf("Grow\n");
	    uint64_t t1 = timeSinceEpochMillisec();
	    int x, y, g;
	    while(!this->queue.empty()) {
		//if(this->step % 10 == 0) {
		    //printf("\rStep: %d / %d of %d in groupes / %d", this->step, this->nb_in_groupes, this->nb_pixels, this->queue.size());
		//}

		x = this->queue.front(); this->queue.pop();
		y = this->queue.front(); this->queue.pop();

		// groupe du point
		g = this->queue.front(); this->queue.pop();
		//g = this->tab.at<int>(x, y);

		if(y + 1 < this->hau)
		    this->add_voisin(0, x, y, g, x, y+1);
		if(y - 1 >= 0)
		    this->add_voisin(1, x, y, g, x, y-1);
		if(x + 1 < this->lar)
		    this->add_voisin(2, x, y, g, x+1, y);
		if(x - 1 >= 0)
		    this->add_voisin(3, x, y, g, x-1, y);

		this->step++;
	    }
	    uint64_t t2 = timeSinceEpochMillisec();
	    std::cout << "\nTemps double for: " << t2 - t1 << " ms" << std::endl;
	}

	void add_voisin(int i, int x, int y, int g, int vx, int vy) {
	    // check si le voisin est suffisement proche du groupe du point pour etre ajoute

	    int vg = this->tab.at<int>(vx, vy);

	    // si le voisin n'appartient a aucun groupe
	    if(vg == 0) {
		unsigned char vcol = this->img.at<uchar>(vx, vy);

		double dist = abs(vcol - this->avg_grp_col[g]);

		// si le voisin est proche de la couleur moyenne du groupe
		if(dist <= this->grow_thresh) {
		    // on rajoute le point au groupe
		    this->tab.at<int>(vx, vy) = g;

		    // on met a jour les infos du groupe

		    this->sum_grp_col[g] += vcol;
		    this->avg_grp_col[g] = this->sum_grp_col[g] / ++this->nb_in_groupe[g];
		    this->nb_in_groupes++;

		    this->queue.push(vx);
		    this->queue.push(vy);
		    this->queue.push(g);
		}

	    }
	}

	void merge(void) {
	    uint64_t t1 = timeSinceEpochMillisec();
	    std::cout << "=================================" << std::endl;
	    std::cout << "MERGE" << std::endl;
	    int g;
	    for (int x = 0; x < this->lar; ++x) {
		for (int y = 0; y < this->hau; ++y) {
		    g = this->tab.at<int>(x, y);

		    if(y + 1 < this->hau)
			this->merge_voisin(0, x, y, g, x, y+1);
		    if(y - 1 >= 0)
			this->merge_voisin(1, x, y, g, x, y-1);
		    if(x + 1 < this->lar)
			this->merge_voisin(2, x, y, g, x+1, y);
		    if(x - 1 >= 0)
			this->merge_voisin(3, x, y, g, x-1, y);
		}
	    }
	    uint64_t t2 = timeSinceEpochMillisec();
	    std::cout << "Temps: " << t2 - t1 << " ms" << std::endl;

	}

	void merge_voisin(int i, int x, int y, int g, int vx, int vy) {
	    int vg = this->val_groupe[this->tab.at<int>(vx, vy)];

	    if(vg != g) {
		// les deux pixels sont dans des groupes differents donc on tente le merge de groupes
		int dist = abs(this->avg_grp_col[vg] - this->avg_grp_col[g]);
		if(pow(dist, 2) <= this->merge_thresh) {
		    this->sum_grp_col[g] += this->sum_grp_col[vg];
		    this->sum_grp_col[vg] = 0;

		    this->nb_in_groupe[g] += this->nb_in_groupe[vg];
		    this->nb_in_groupe[vg] = 0;

		    this->avg_grp_col[g] = this->sum_grp_col[g] / this->nb_in_groupe[g];

		    for (int i = 1; i < this->nb_germes+1; ++i) {
			if(this->val_groupe[i] == vg) {
			    this->val_groupe[i] = g;
			}
		    }
		}
	    }
	}

	// ==========================================================================
	Mat create_seg_img(void) {
	    // on recree une image en colorant chaque pixels appartenant au meme groupe de la meme couleur aleatoire
	    //std::cout << "=================================" << std::endl;
	    //printf("create_seg_imge\n");
	    //uint64_t t1 = timeSinceEpochMillisec();
	    int cols[(this->nb_germes+1)*3];

	    std::default_random_engine generator;
	    std::uniform_int_distribution<int> distribution(50,200);

	    // creation de couleurs aleatoires pour chaque groupe pour colorer l'image
	    for (int i = 3; i < (this->nb_germes+1) * 3; i+=3) {
		cols[i]   = distribution(generator);
		cols[i+1] = distribution(generator);
		cols[i+2] = distribution(generator);
	    }

	    Size img_size = this->img.size();
	    Mat img_seg;
	    img_seg.create(img_size, CV_8UC3);


	    int g;
	    if(this->display_true_col) {
		for (int x = 0; x < this->lar; ++x) {
		    for (int y = 0; y < this->hau; ++y) {
			g = this->val_groupe[this->tab.at<int>(x, y)];

			img_seg.at<Vec3b>(x, y)[0] = (uchar) this->avg_grp_col[g];
			img_seg.at<Vec3b>(x, y)[1] = (uchar) this->avg_grp_col[g];
			img_seg.at<Vec3b>(x, y)[2] = (uchar) this->avg_grp_col[g];
		    }
		}
	    } else {
		for (int x = 0; x < this->lar; ++x) {
		    for (int y = 0; y < this->hau; ++y) {
			g = this->val_groupe[this->tab.at<int>(x, y)];

			if(g > 0) {
			    img_seg.at<Vec3b>(x, y)[0] = (uchar) cols[g*3];
			    img_seg.at<Vec3b>(x, y)[1] = (uchar) cols[g*3+1];
			    img_seg.at<Vec3b>(x, y)[2] = (uchar) cols[g*3+2];
			}
		    }
		}

	    }

	    //uint64_t t2 = timeSinceEpochMillisec();
	    //std::cout << "Temps double for: " << t2 - t1 << " ms" << std::endl;

	    return img_seg;
	}

	void trace_contour(Mat img_seg) {
	    //std::cout << "=================================" << std::endl;
	    //printf("Contour\n");
	    //uint64_t t1 = timeSinceEpochMillisec();
	    Mat edge1, edge2, kernel1, kernel2, img_seg_gray;

	    cv::Mat kernelH(3, 3, CV_32F);
	    kernelH.at<float>(0,0) = 1.0f;
	    kernelH.at<float>(0,1) = 1.0f;
	    kernelH.at<float>(0,2) = 1.0f;

	    kernelH.at<float>(1,0) = 0.0f;
	    kernelH.at<float>(1,1) = 0.0f;
	    kernelH.at<float>(1,2) = 0.0f;

	    kernelH.at<float>(2,0) = -1.0f;
	    kernelH.at<float>(2,1) = -1.0f;
	    kernelH.at<float>(2,2) = -1.0f;

	    cv::Mat kernelV(3, 3, CV_32F);
	    kernelV.at<float>(0,0) = 1.0f;
	    kernelV.at<float>(1,0) = 1.0f;
	    kernelV.at<float>(2,0) = 1.0f;

	    kernelV.at<float>(0,1) = 0.0f;
	    kernelV.at<float>(1,1) = 0.0f;
	    kernelV.at<float>(2,1) = 0.0f;

	    kernelV.at<float>(0,2) = -1.0f;
	    kernelV.at<float>(1,2) = -1.0f;
	    kernelV.at<float>(2,2) = -1.0f;

	    cvtColor(img_seg, img_seg_gray, COLOR_BGR2GRAY);

	    cv::filter2D(img_seg_gray, edge1, -1, kernelH);
	    cv::filter2D(img_seg_gray, edge2, -1, kernelV);

	    Mat res;
	    cv::add(edge1, edge2, res);

	    img_seg.forEach<Pixel> ( [res](Pixel &pixel, const int * position) -> void {
		    if(res.at<uchar>(position[0], position[1]) != 0) {
			pixel.x = (uchar) 255;
			pixel.y = (uchar) 255;
			pixel.z = (uchar) 255;
		    }
	      }
	    );

	    //uint64_t t2 = timeSinceEpochMillisec();
	    //std::cout << "Temps double for: " << t2 - t1 << " ms" << std::endl;
	}

	~Segmentor(){
	    delete[] this->avg_grp_col;
	    delete[] this->nb_in_groupe;
	}
};

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

const char* keys =
{
    "{help h||}{@image |fruits.jpg|input image name}"
};

int main( int argc, const char** argv ) {
    Mat base_image, img, gray, frame;
    CommandLineParser parser(argc, argv, keys);
    string filename = parser.get<string>(0);

    if(filename == "cam") {
	VideoCapture cap;
	cap.open(0);
	if ( ! cap.isOpened() ) {
	    cout << "--(!)Error opening video capture\n";
	    return -1;
	}

	double dWidth = cap.get(CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
	double dHeight = cap.get(CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video

	cout << "Resolution of the video : " << dWidth << " x " << dHeight << endl;

	Segmentor seg = Segmentor();
	while ( cap.read(frame) ) {
	    if( frame.empty() )
	    {
		cout << "--(!) No captured frame -- Break!\n";
		break;
	    }

	    std::cout << "\nlllllllllllllllllllllllllllllllllllllllllll\n" << std::endl;

	    uint64_t t1 = timeSinceEpochMillisec();

	    //bilateralFilter(frame, img, 10, 25, 25);
	    GaussianBlur(frame, img, Size(15, 15), 15, 15);
	    //blur(frame, img, Size(3, 3));

	    cvtColor(img, gray, COLOR_BGR2GRAY);

	    uint64_t t2 = timeSinceEpochMillisec();
	    std::cout << "Temps prep: " << t2 - t1 << " ms" << std::endl;

	    // img, nb_seeds, merge_thresh, grow_thresh, display_true_col
	    Mat img_seg = seg.segmentation(gray, 101, 150, 30, false);

	    uint64_t t3 = timeSinceEpochMillisec();
	    std::cout << endl << "Temps Total: " << t3 - t1 << " ms" << std::endl;

	    //imshow("fdfdf", gray);
	    //waitKey(0);

	    imshow("fdfdf", img_seg);
	    waitKey(1);
	    //sleep(1000);
	}
    } else if (filename == "tPoseMan.mp4")
	{
		VideoCapture capture("tPoseMan.mp4");

		String file = "ResultVideo.avi";
		Size S(640, 360);
		int ex = VideoWriter::fourcc('X','V','I','D');
		VideoWriter out(file, ex, capture.get(CAP_PROP_FPS), S);
		while (capture.isOpened())
		{
			/*frameSkip++;
			if (frameSkip >= MAXFRAMESKIP)
			{
				frameSkip = 0;
			}*/
			capture >> base_image;
			/*if (frameSkip != 0)
			{
				continue;
			}*/
			
			
			uint64_t t1 = timeSinceEpochMillisec();
			
			if (base_image.empty())
			{
				return -1;
			}
			//filter ici
			GaussianBlur(base_image, img, Size(15, 15), 15, 15);
			cvtColor(img, gray, COLOR_BGR2GRAY);
			
			Segmentor seg = Segmentor();

			Mat img_seg = seg.segmentation(gray, 2001, 150, 15, false);


			uint64_t t3 = timeSinceEpochMillisec();
			std::cout << "Temps Total: " << t3 - t1 << " ms" << std::endl;

			imshow("Result", gray);
			//write result
			out.write(gray);
		}
		out.release();
	} else {
	uint64_t t1 = timeSinceEpochMillisec();

	base_image = imread(samples::findFile(filename), IMREAD_COLOR);
	if(base_image.empty())
	{
	    printf("Cannot read image file: %s\n", filename.c_str());
	    return -1;
	}

	//bilateralFilter(base_image, img, 5, 15, 15);
	GaussianBlur(base_image, img, Size(15, 15), 15, 15);

	cvtColor(img, gray, COLOR_BGR2GRAY);

	//imshow("fdfdf", gray);
	//waitKey(0);

	Segmentor seg = Segmentor();

	//uint64_t t2 = timeSinceEpochMillisec();
	//std::cout << "Temps prepar: " << t2 - t1 << " ms" << std::endl;

	// img, nb_seeds, merge_thresh, grow_thresh
	Mat img_seg = seg.segmentation(gray, 2001, 150, 15, false);

	uint64_t t3 = timeSinceEpochMillisec();
	std::cout << "Temps Total: " << t3 - t1 << " ms" << std::endl;

	imshow("Result", img_seg);
	waitKey(0);
    }
}
