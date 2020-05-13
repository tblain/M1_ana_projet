#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <queue>
#include <random>
#include <typeinfo>

#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

typedef cv::Point3_<uint8_t> Pixel;

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

	// pour chaque groupe: le nb de pixels a l'interieur
	int *nb_in_groupe;

	// tableau contenant pour chaque point l'index du groupe auquel il appartient ou alors 0
	Mat tab;

	// tableau contenant pour chaque groupe la couleur moyenne
	double *avg_grp_col;

	std::queue<int> queue;

	int step;
	int nb_pixels;

	// =========== Methodes ================================================

	// ==========================================================================
	int* create_germes_regular(void) {
	    printf("create_germes_random\n");
	    int* germes = new int[2 * (this->nb_germes+1)];
	    int dist_between_pts = this->nb_pixels / this->nb_germes;
	    std::cout << "Dist: " << dist_between_pts << " / " << this->lar << std::endl;

	    for (int i = 2; i < (this->nb_germes+1)*2; i+=2) {
		int n = dist_between_pts * i / 2;
		int x = n % this->lar;
		int y = n / this->lar;

		germes[i] = x;
		germes[i+1] = y;
		//std::cout << "Point: " << n << " : " << x << " / " << y << std::endl;
	    }

	    return germes;
	}

	// ==========================================================================
	int* create_germes_random(void) {
	    printf("create_germes_random\n");
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
	    printf("init_segmentation\n");
	    int* germes = this->create_germes_regular();

	    Size img_size = img.size();
	    this->tab.create(img_size, CV_64F);
	    this->avg_grp_col = new double[this->nb_germes+1];
	    this->nb_in_groupe = new int[this->nb_germes+1];
	    this->nb_in_groupes = 0;

	    int x, y;
	    for (int i = 2; i < (this->nb_germes+1)*2; i+=2) {
		x = germes[i];
		y = germes[i+1];
		//printf("for %d: %d %d\n", (double)i/2, x, y);

		this->nb_in_groupes++;
		this->tab.at<double>(x, y) = (double) i/2;
		this->avg_grp_col[i/2] = this->img.at<uchar>(x, y);
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
	    printf("pre_seg_dist_calcul\n");

	}

	// ==========================================================================
	Mat segmentation(Mat img, int nb_germes) {
	    printf("Seg\n");
	    this->img = img;
	    this->nb_germes = nb_germes;

	    this->lar = img.size[0];
	    this->hau = img.size[1];
	    this->nb_pixels = this->hau * this->lar;
	    printf("Size: %d %d\n", this->lar, this->hau);

	    this->init_segmentation();
	    this->pre_seg_dist_calcul();

	    this->step = 0;

	    this->grow();
	    return this->create_seg_img();
	}

	// ==========================================================================
	void grow(void) {
	    printf("Grow\n");
	    int x, y;
	    double g;
	    while(!this->queue.empty()) {
		if(this->step % 10000 == 0) {
		    //printf("Step: %d / %d of %d in groupes / %d\n", this->step, this->nb_in_groupes, this->nb_pixels, this->queue.size());
		}

		x = this->queue.front(); this->queue.pop();
		y = this->queue.front(); this->queue.pop();
		//printf("X: %d / Y: %d\n", x, y);

		// groupe du point
		g = this->tab.at<double>(x, y);
		//std::cout << "Groupe: " << g << std::endl;

		this->add_voisin(0, x, y, g, x, y+1);
		this->add_voisin(1, x, y, g, x, y-1);
		this->add_voisin(2, x, y, g, x+1, y);
		this->add_voisin(3, x, y, g, x-1, y);

		this->step++;
	    }

	}

	void add_voisin(int i, int x, int y, int g, int vx, int vy) {
	    // check si le voisin est suffisement proche du groupe du point pour etre ajoute

	    double vg;
	    unsigned char vcol;

	    // on verifie que le voisin est bien dans l'image
	    if(vx >= 0 && vy >= 0 && vx < this->lar && vy < this->hau) {
		vg = this->tab.at<double>(vx, vy);

		// si le voisin n'appartient a aucun groupe
		if(vg == 0) {
		    //std::cout << "Dedans: " << vg << std::endl;
		    vcol = this->img.at<uchar>(vx, vy);

		    // si le voisin est proche de la couleur moyenne du groupe
			//printf("Col: %d / diff: %d\n", vcol, abs(vcol - this->avg_grp_col[g]));
		    double dist = abs(vcol - this->avg_grp_col[g]);
		    if(pow(dist, 2) <= 20) {

			// on rajoute le point au groupe
			//printf("groupe: %f / diff: %d\n", vg, abs(vcol - this->avg_grp_col[g]));
			this->tab.at<double>(vx, vy) = g;
			//std::cout << "New Groupe: " << this->tab.at<double>(vx, vy) << std::endl;

			// on met a jour les infos du groupe

			this->avg_grp_col[g] = (this->avg_grp_col[g] * this->nb_in_groupe[g] + vcol) / (this->nb_in_groupe[g]+1);
			this->nb_in_groupe[g]++;
			this->nb_in_groupes++;

			this->queue.push(vx);
			this->queue.push(vy);
		    }

		} else if(true && vg != g) {
		    // les deux pixels sont des des groupes differents donc on tente le merge
		    double dist = abs(this->avg_grp_col[(int)vg] - this->avg_grp_col[(int)g]);
		    if(pow(dist, 2) <= 10) {
			//std::cout << "Remplace: " << vg << " / " << g << std::endl;

			double avg_g = this->avg_grp_col[g] * this->nb_in_groupe[g];
			double avg_vg = this->avg_grp_col[(int)vg] * this->nb_in_groupe[(int)vg];

			this->avg_grp_col[g] = (avg_g + avg_vg) / (this->nb_in_groupe[g] + this->nb_in_groupe[(int)vg]);
			this->nb_in_groupe[g] += this->nb_in_groupe[(int) vg];
			this->nb_in_groupe[(int) vg] = 0;

			//this->tab.forEach<Pixel> ( [vg, g](Pixel &pixel, const int * position) -> void {
			    //if(pixel.x == vg) {
				//pixel.x = g;
			    //}
			  //}
			//);
			//std::cout << "f" << std::endl;

			//for (int i = 0;i < this->lar; ++i) {
			    //for (int j = 0; j < this->hau; ++j) {
				//if(this->tab.at<double>(i, j) == vg) {
				    //this->tab.at<double>(i, j) = g;
				//}

			    //}
			//}
		    }
		}
	    }

	}

	// ==========================================================================
	Mat create_seg_img(void) {
	    // on recree une image en colorant chaque pixels appartenant au meme groupe de la meme couleur aleatoire
	    printf("create_seg_imge\n");
	    int cols[(this->nb_germes+1)*3];

	    std::default_random_engine generator;
	    std::uniform_int_distribution<int> distribution(50,200);

	    // creation de couleurs aleatoires pour chaque groupe pour colorer l'image
	    for (int i = 3; i < (this->nb_germes+1) * 3; i+=3) {
		cols[i] = distribution(generator);
		cols[i+1] = distribution(generator);
		cols[i+2] = distribution(generator);
	    }

	    int g;
	    Size img_size = this->img.size();
	    Mat img_seg;
	    img_seg.create(img_size, CV_8UC3);
	    std::cout << "Mat: " << img_seg.depth() << std::endl;

	    for (int x = 0; x < this->lar; ++x) {
		for (int y = 0; y < this->hau; ++y) {
		    g = (int) this->tab.at<double>(x, y);
		    img_seg.at<Vec3b>(x, y)[0] = (double) cols[g*3];
		    img_seg.at<Vec3b>(x, y)[1] = (double) cols[g*3+1];
		    img_seg.at<Vec3b>(x, y)[2] = (double) cols[g*3+2];
		}
	    }

	    std::cout << "FINISH" << std::endl;
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
    Mat base_image, img, gray;
    CommandLineParser parser(argc, argv, keys);
    string filename = parser.get<string>(0);

    base_image = imread(samples::findFile(filename), IMREAD_COLOR);
    if(base_image.empty())
    {
        printf("Cannot read image file: %s\n", filename.c_str());
        return -1;
    }

    bilateralFilter(base_image, img, 10, 75, 75);

    cvtColor(img, gray, COLOR_BGR2GRAY);

    //imshow("fdfdf", base_image);
    //waitKey(0);

    //imshow("fdfdf", img);
    //waitKey(0);
    //printf("%s\n", type2str(gray.type()).c_str());
    //printf("%s\n", type2str(image.type()).c_str());

    Segmentor seg = Segmentor();
    Mat img_seg = seg.segmentation(gray, 1001);
    imshow("fdfdf", img_seg);
    waitKey(0);
}
