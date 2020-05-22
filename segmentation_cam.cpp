#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"

#include <queue>
#include <random>
#include <typeinfo>
#include <chrono>

#include <stdio.h>
#include <iostream>
#include <stdlib.h>     //for using the function sleep
#include <time.h>
#include <unistd.h>

using namespace cv;
using namespace std;
using namespace std::chrono;

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

	std::queue<int> queue;

	int step;
	int nb_pixels;

	int merge_thresh;
	int grow_thresh;

	// =========== Methodes ================================================

	// ==========================================================================
	int* create_germes_regular(void) {
	    //std::cout << "=================================" << std::endl;
	    //printf("create_germes_random\n");

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
	int* create_germes_random(void) {
	    //std::cout << "=================================" << std::endl;
	    //printf("create_germes_random\n");
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
	    //std::cout << "=================================" << std::endl;
	    //printf("init_segmentation\n");
	    int* germes = this->create_germes_regular();

	    Size img_size = img.size();
	    this->tab.create(img_size, CV_32S);
	    this->avg_grp_col = new double[this->nb_germes+1];
	    this->nb_in_groupe = new int[this->nb_germes+1];
	    this->nb_in_groupes = 0;

	    this->groupes = new std::vector<int>[this->nb_germes+1];
	    this->val_groupe = new int[this->nb_germes+1];

	    this->val_groupe[0] = 0;
	    for (int i = 1; i < nb_germes+1; ++i) {
		this->val_groupe[i] = i;
	    }


	    int x, y;
	    // on instancie les seeds
	    for (int i = 2; i < (this->nb_germes+1)*2; i+=2) {
		x = germes[i];
		y = germes[i+1];

		this->nb_in_groupes++;
		this->tab.at<int>(x, y) = i/2;
		this->avg_grp_col[i/2] = (double) this->img.at<uchar>(x, y);
		//std::cout << this->img.at<uchar>(5, 5) << std::endl;
		this->nb_in_groupe[i/2] = 1;

		this->queue.push(x);
		this->queue.push(y);
	    }

	    delete[] germes;

	    // TODO: voir si ca fonctionne comme ca
	    //this.queue = new Queue();
	}
	// ==========================================================================
	void pre_seg_dist_calcul(void) {
	    //printf("pre_seg_dist_calcul\n");

	}

	// ==========================================================================
	Mat segmentation(Mat img, int nb_germes, int merge_thresh, int grow_thresh) {
	    //std::cout << "=================================" << std::endl;
	    //printf("Seg\n");
	    this->img = img;
	    this->nb_germes = nb_germes;

	    this->lar = img.size[0];
	    this->hau = img.size[1];

	    this->merge_thresh = merge_thresh;
	    this->grow_thresh = grow_thresh;
	    this->nb_pixels = this->hau * this->lar;
	    //printf("Size: %d %d\n", this->lar, this->hau);

	    this->init_segmentation();
	    this->pre_seg_dist_calcul();

	    this->step = 0;

	    this->grow();

	    this->merge();
	    return this->create_seg_img();
	}

	// ==========================================================================
	void grow(void) {
	    //std::cout << "=================================" << std::endl;
	    //printf("Grow\n");
	    //uint64_t t1 = timeSinceEpochMillisec();
	    int x, y;
	    int g;
	    while(!this->queue.empty()) {
		//if(this->step % 10000 == 0) {
		    //printf("\rStep: %d / %d of %d in groupes / %d", this->step, this->nb_in_groupes, this->nb_pixels, this->queue.size());
		//}

		x = this->queue.front(); this->queue.pop();
		y = this->queue.front(); this->queue.pop();

		// groupe du point
		g = this->tab.at<int>(x, y);

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
	    //uint64_t t2 = timeSinceEpochMillisec();
	    //std::cout << "\nTemps double for: " << t2 - t1 << " ms" << std::endl;

	}

	void add_voisin(int i, int x, int y, int g, int vx, int vy) {
	    // check si le voisin est suffisement proche du groupe du point pour etre ajoute
		std::cout << "type: " << this->img.at<uchar>(5, 5) << std::endl;

	    int vg = this->tab.at<int>(vx, vy);

	    // si le voisin n'appartient a aucun groupe
	    if(vg == 0) {
		uchar vcol = this->img.at<uchar>(5, 5);

		// si le voisin est proche de la couleur moyenne du groupe
		double dist = abs(vcol - this->avg_grp_col[g]);

		//std::cout << vcol << " / dist: " << dist << " / " << this->img.at<double>(vx, vy) << " / " << x << " / " << y << std::endl;
		if(dist <= this->grow_thresh) {

		    // on rajoute le point au groupe
		    this->tab.at<int>(vx, vy) = g;

		    // on met a jour les infos du groupe

		    this->avg_grp_col[g] = (this->avg_grp_col[g] * this->nb_in_groupe[g] + vcol) / (this->nb_in_groupe[g]+1);
		    this->nb_in_groupe[g]++;
		    this->nb_in_groupes++;

		    this->queue.push(vx);
		    this->queue.push(vy);
		}

	    }
	}

	void merge(void) {
	    //uint64_t t1 = timeSinceEpochMillisec();
	    int g;
	    //std::cout << "=================================" << std::endl;
	    //std::cout << "MERGE" << std::endl;
	    for (int x = 0; x < this->lar; ++x) {
		//std::cout << x << std::endl;
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
	    //uint64_t t2 = timeSinceEpochMillisec();
	    //std::cout << "Temps: " << t2 - t1 << " ms" << std::endl;

	}

	void merge_voisin(int i, int x, int y, int g, int vx, int vy) {
	    int vg = this->val_groupe[this->tab.at<int>(vx, vy)];

	    if(vg != g) {
		// les deux pixels sont dans des groupes differents donc on tente le merge de groupes
		int dist = abs(this->avg_grp_col[vg] - this->avg_grp_col[g]);
		if(pow(dist, 2) <= this->merge_thresh) {

		    double avg_g = this->avg_grp_col[g] * this->nb_in_groupe[g];
		    double avg_vg = this->avg_grp_col[vg] * this->nb_in_groupe[vg];

		    this->avg_grp_col[g] = (avg_g + avg_vg) / (this->nb_in_groupe[g] + this->nb_in_groupe[vg]);
		    this->nb_in_groupe[g] += this->nb_in_groupe[vg];
		    this->nb_in_groupe[vg] = 0;

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
	    int cols[(this->nb_germes+1)*3];

	    std::default_random_engine generator;
	    std::uniform_int_distribution<int> distribution(50,200);

	    // creation de couleurs aleatoires pour chaque groupe pour colorer l'image
	    for (int i = 3; i < (this->nb_germes+1) * 3; i+=3) {
		//std::cout << "cG: " << i << " / " << this->val_groupe[i] << std::endl;
		cols[i] = distribution(generator);
		cols[i+1] = distribution(generator);
		cols[i+2] = distribution(generator);
	    }

	    Size img_size = this->img.size();
	    Mat img_seg;
	    img_seg.create(img_size, CV_8UC3);

	    //uint64_t t1 = timeSinceEpochMillisec();

	    int g;
	    for (int x = 0; x < this->lar; ++x) {
		for (int y = 0; y < this->hau; ++y) {
		    g = this->val_groupe[this->tab.at<int>(x, y)];
		    //std::cout << "VG: " << g << " / " << this->tab.at<int>(x, y) << std::endl;
		    //img_seg.at<Vec3b>(x, y)[0] = (uchar) cols[g*3];
		    //img_seg.at<Vec3b>(x, y)[1] = (uchar) cols[g*3+1];
		    //img_seg.at<Vec3b>(x, y)[2] = (uchar) cols[g*3+2];

		    img_seg.at<Vec3b>(x, y)[0] = (uchar) this->avg_grp_col[g];
		    img_seg.at<Vec3b>(x, y)[1] = (uchar) this->avg_grp_col[g];
		    img_seg.at<Vec3b>(x, y)[2] = (uchar) this->avg_grp_col[g];
		}
	    }

	    //uint64_t t2 = timeSinceEpochMillisec();
	    //std::cout << "Temps double for: " << t2 - t1 << " ms" << std::endl;

	    //img_seg.forEach<Pixel> ( [&cols, this](Pixel &pixel, const int * position) -> void {
		    //int g = this->val_groupe[this->tab.at<int>(position[0], position[1])];
		    //pixel.x = (uchar) cols[g*3];
		    //pixel.y = (uchar) cols[g*3+1];
		    //pixel.z = (uchar) cols[g*3+2];
	      //}
	    //);

	    //uint64_t t2 = timeSinceEpochMillisec();
	    //std::cout << "Temps double for: " << t2 - t1 << " ms" << std::endl;

	    //std::cout << "FINISH" << std::endl;
	    return img_seg;
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

	uint64_t t1 = timeSinceEpochMillisec();

	//bilateralFilter(frame, img, 15, 45, 45);
	//GaussianBlur(frame, img, Size(15, 15), 0, 0);
	blur(frame, img, Size(3, 3));

	cvtColor(img, gray, COLOR_BGR2GRAY);
	std::cout << "Type: " << type2str(gray.type()).c_str() << std::endl;

	// img, nb_seeds, merge_thresh, grow_thresh
	Mat img_seg = seg.segmentation(gray, 21, 100, 1);

	uint64_t t2 = timeSinceEpochMillisec();
	std::cout << "Temps double for: " << t2 - t1 << " ms" << std::endl;

	//imshow("fdfdf", gray);
	//waitKey(1);

	imshow("fdfdf", img_seg);
	waitKey(0);
	//sleep(1000);

    }
    return 0;

    //imshow("fdfdf", base_image);
    //waitKey(0);

    //imshow("fdfdf", img);
    //waitKey(0);

}
