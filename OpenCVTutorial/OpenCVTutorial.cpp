#include "stdafx.h"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;
//Displays a picture in a window and waits for the user to press a key
int tutorial1()
{
    string imageName("MyPic.jpg"); // by default
    Mat image;
    image = imread(imageName.c_str(), IMREAD_COLOR); // Read the file
    if( image.empty() )                      // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    cvNamedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
    imshow( "Display window", image );                // Show our image inside it.
    waitKey(0); // Wait for a keystroke in the window
	cvDestroyWindow("Display window"); 
    return 0;
}
//Displays a video but stops when the user presses the esc key 
int tutorial2()
{
	VideoCapture cap("Jump To Hyperspace.mp4"); 
	if (!cap.isOpened())
	{
		cout << "Cannot Open Video file" << endl; 
		system("pause"); 
		return -1; 
	}
	double fps = cap.get(CV_CAP_PROP_FPS);
	cout << "Frames per second " << fps << endl; 
	cvNamedWindow("MyVideo",CV_WINDOW_AUTOSIZE);

	//Loads and displays each frame of the video 
	while(1)
	{
		Mat frame; 

		bool bSuccess = cap.read(frame); 

		if (!bSuccess)
		{
			cout << "Cannot read from video file" << endl; 
			system("pause");
			return -1;
		}

		imshow("MyVideo",frame); 
		if(waitKey(30) == 27)
		{
			cout << "Esc key pressed by user" << endl;
			return -1; 
		}
	}
	cvDestroyWindow("MyVideo"); 
	return 0;


}
//Writing an image to a file 
int tutorial3()
{
	Mat img(650, 600, CV_16UC3, Scalar(0,50000, 50000)); 
	if (img.empty())
	{
		cout << "ERROR: Image cannot be loaded" << endl; 
		system("pause"); 
		return -1; 
	}

	vector<int> compression_params; 
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY); 
	compression_params.push_back(98); 

	bool bSuccess = imwrite("C:/TestImage.jpg", img, compression_params); 
	if (!bSuccess)
	{
		cout << "Error: Failed to save image" << endl; 
		system("pause"); 
		return -1; 
	}
	 cvNamedWindow("MyWindow", CV_WINDOW_AUTOSIZE);
     imshow("MyWindow", img);

	 waitKey(0); 

	 cvDestroyWindow("MyWindow");

	 return 0;


}
//3 types of Image Filtering: Errode,Dilate, and Invert
int tutorial4()
{
	//Display Original Image 
	    IplImage* img = cvLoadImage("baboon.jpg"); 
        cvNamedWindow("MyWindow");
        cvShowImage("MyWindow", img);
		waitKey(0);
		cvDestroyWindow("MyWindow");

	//Errode Image
		cvErode(img, img, 0, 2); 
		cvNamedWindow("Erroded"); 
		cvShowImage("Erroded",img); 
		waitKey(0); 
		cvDestroyWindow("Erroded"); 
	
	//Dilated Image
		cvDilate(img, img, 0, 2); 
		cvNamedWindow("Erroded"); 
		cvShowImage("Erroded",img); 
		waitKey(0); 
		cvDestroyWindow("Erroded"); 
	
	//Invert Image
		cvNot(img,img); 
		cvNamedWindow("Inverted"); 
		cvShowImage("Inverted",img); 
		waitKey(0); 
		cvDestroyWindow("Inverted"); 

	//Clean Up
		cvReleaseImage(&img);  

	return 0; 
}
//Shows how to change brightness,contrast of a video, equalize the historgram, and Smooth(Blur) images 
int tutorial5()
{
	Mat img = imread("board.jpg"); 

	if (img.empty())
	{
		cout << "Error reading image" << endl; 
		return -1; 
	}

	//Changing Brightness of an image, Video is similar but you use a loop and brighten/darken each frame
	Mat imgH = img + Scalar(75,75,75); //Can also use convertTo fxn, 75 added to all channels RGB for every pixel

	Mat imgL;
	img.convertTo(imgL,-1,1,-75); //Can also use img + Scalar, mult each pixel by 1 and add -75 then store in imgL

	cvNamedWindow("Original",CV_WINDOW_AUTOSIZE); 
	cvNamedWindow("Brighter", CV_WINDOW_AUTOSIZE); 
	cvNamedWindow("Darker", CV_WINDOW_AUTOSIZE);

	imshow("Original", img); 
	imshow("Brighter", imgH); 
	imshow("Darker", imgL); 
	waitKey(0); 

	cvDestroyAllWindows(); 

	//Changing Contrast of an Image
	img.convertTo(imgH, -1,2,0); //Every pixel is multiplied by 2 and has 0 added and then stored in imgH
	img.convertTo(imgL,-1,0.5,0); 

	cvNamedWindow("Original",CV_WINDOW_AUTOSIZE); 
	cvNamedWindow("Brighter", CV_WINDOW_AUTOSIZE); 
	cvNamedWindow("Darker", CV_WINDOW_AUTOSIZE);

	imshow("Original", img); 
	imshow("Brighter", imgH); 
	imshow("Darker", imgL); 
	waitKey(0); 

	cvDestroyAllWindows(); 

	//Histogram Equialization (GreyScale), Color requires to split into channels, use the scheme YCrCb format, and equalize each channel
	cvtColor(img, img, CV_BGR2GRAY); //Change to greyscale
	Mat img_hist_equalized;
	equalizeHist(img, img_hist_equalized); 

	cvNamedWindow("Original"); 
	cvNamedWindow("Equalized"); 

	imshow("Original", img); 
	imshow("Equalized", img_hist_equalized); 
	waitKey(0); 
	cvDestroyAllWindows(); 

	//Blur(Smooth) images 
	//Homoginzied
	namedWindow("Original", CV_WINDOW_AUTOSIZE);
	namedWindow("Smooth", CV_WINDOW_AUTOSIZE); 
	imshow("Original",img); 

	Mat dst; 
	char zBuffer[35]; 


	for(int i =1; i<31; i = i+2)
	{
		_snprintf_s(zBuffer, 35,"Kernel Size : %d x %d", i, i); //Copy text to buffer
		blur(img,dst,Size(i,i)); //blur the image with kernel size i,i 
		putText( dst, zBuffer, Point( img.cols/4, img.rows/8), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255) ); //Put text in zBuff into dst image
		imshow("Smooth", dst); 
		int c = waitKey(2000); 
		if(c == 27)
		{
			cvDestroyAllWindows();
			return(0); 
		}


	}
	dst = Mat::zeros(img.size(), img.type()); 
	 _snprintf_s(zBuffer, 35,"Press Any Key to Exit"); //Copy text to buffer
	 putText( dst, zBuffer, Point( img.cols/4,  img.rows / 2), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255) ); //put text onto image
	 imshow( "Smoothed Image", dst );
	 waitKey(0);
	 cvDestroyAllWindows(); 

	 //Guassian Smoothing, same as above but instead of blur use   GaussianBlur w/extra params 0,0
	 //Median Smoothing, same as above but with function medianBlur(src,dest,i)
	 //Bilateral Smoothing, same as above but with function bilateralFilter( src, dst, i, i, i)



	return(0); 

}

//Shows how to create a trackbar that allows user to change brightness and contrast of image
int tutorial6()
{
	Mat src = imread("fruits.jpg"); 
	if(!src.data) 
	{
		cout << "Failed to read img" << endl; 
		return -1; 
	}

	cvNamedWindow("My Window"); 

	int iSliderValue = 50; 
	createTrackbar("Brightness", "My Window", &iSliderValue,100); 

	int iSliderValue2 = 50; 
	createTrackbar("Contrast", "My Window", &iSliderValue2, 100); 

	//Loop that allows the user to use trackbars until they hit esc
	while(true)
	{
		Mat dst; 
		int iBrightness = iSliderValue - 50; 
		double dContrast = iSliderValue2/50.0; 
		src.convertTo(dst, -1, dContrast, iBrightness); 

		imshow("My Window", dst); 

		int iKey = waitKey(50); 

		if(iKey == 27)
		{
			break; 
		}

	}

	cvDestroyWindow("My Window"); 
	return 0; 
}

//Helper function for tutorial 7, tells what message to send on various events 
void CallBackFunc1(int event, int x, int y, int flags, void* userdata) 
{
	if (event == EVENT_LBUTTONDOWN) 
	{
		cout << "Left button clicked at position (" << x << "," << y << ")" << endl; 
	}
	else if (event == EVENT_RBUTTONDOWN) 
	{
		cout << "Left button clicked at position(" << x << "," << y << ")" << endl; 
	}
	else if (event == EVENT_MBUTTONDOWN) 
	{
		cout << "Middle button clicked at position(" << x << "," << y << ")" << endl; 
	}
	else if ( event == EVENT_MOUSEMOVE )
    {
        cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl;
	}
}
//Helper function for tutorial 7, tells what message to send on various events 
void CallBackFunc2(int event, int x, int y, int flags, void* userdata) 
{
	if (flags == EVENT_FLAG_CTRLKEY + EVENT_FLAG_LBUTTON) 
	{
		cout << "Left button clicked while holding ctrl button at position (" << x << "," << y << ")" << endl; 
	}
	else if (flags == EVENT_FLAG_CTRLKEY + EVENT_FLAG_RBUTTON) 
	{
		cout << "Right button clicked while holding ctrl button at position(" << x << "," << y << ")" << endl; 
	}
	else if ( event == EVENT_MOUSEMOVE && flags == EVENT_FLAG_ALTKEY)
    {
        cout << "Mouse move over the window while holding alt - position (" << x << ", " << y << ")" << endl;
	}
} 
//Tracks mouse movment and clicks then displays the appropriate message 
int tutorial7()
{
	Mat img = imread("logo.png"); 
	if ( img.empty() ) 
    { 
		cout << "Error loading the image" << endl;
        return -1; 
    }
	//Detect mouse movement with no key push
	cvNamedWindow("MyWindow", 1); 
	setMouseCallback("MyWindow",CallBackFunc1, NULL); 
	imshow("MyWindow",img); 
	waitKey(0); 
	cvDestroyWindow("MyWindow"); 

	//Detect mouse movement with key push
	cvNamedWindow("MyWindow", 1); 
	setMouseCallback("MyWindow",CallBackFunc2, NULL); 
	imshow("MyWindow",img); 
	waitKey(0); 
	cvDestroyWindow("MyWindow");
	return 0; 

}
//Allows the user to rotate an image using a trackbar 
int tutorial8()
{
	Mat imgOriginal = imread("stuff.jpg"); 
	const char* pzOriginalImage = "Original Image"; 
	cvNamedWindow(pzOriginalImage, CV_WINDOW_AUTOSIZE); 
	imshow(pzOriginalImage,imgOriginal);

	//Rotate Image 
	const char* pzRotatedImage = "Rotated Image";
    namedWindow( pzRotatedImage, CV_WINDOW_AUTOSIZE );

	int iAngle = 180; 
	createTrackbar("Angle", pzRotatedImage,&iAngle, 360); 

	int iImageHeight = imgOriginal.rows/2; 
	int iImageWidth = imgOriginal.cols/2; 

	while(true)
	{
		Mat matRotation = getRotationMatrix2D(Point(iImageHeight,iImageWidth), (iAngle-180),1); 
		Mat imgRotated; 
		warpAffine(imgOriginal,imgRotated, matRotation, imgOriginal.size()); 

		imshow(pzRotatedImage, imgRotated); 

		int iRet = waitKey(30); 
		if (iRet == 27)
		{
			break; 
		}

	}

	//Same for video but do it frame by frame
	return 0; 
}
//Detects an image via webcam and allows the user to adjust hue, saturation, and Value of that image 
int tutorial9()
{
	VideoCapture cap(0); 
	if (!cap.isOpened())
	{
		cout << "Cannot open webcam" << endl; 
		return -1; 
	}

	cvNamedWindow("Control", CV_WINDOW_AUTOSIZE); 

	int iLowH = 0; 
	int iHighH = 179;

	int iLowS = 0; 
	int iHighS=255; 

	int iLowV = 0; 
	int iHighV =255; 

	cvCreateTrackbar("LowH", "Control", &iLowH,179); //Control Hue
	cvCreateTrackbar("HighH", "Control", &iHighH,179);

	cvCreateTrackbar("LowS", "Control", &iLowS,255); //Control Saturation 
	cvCreateTrackbar("HighS", "Control", &iHighS,255);

	cvCreateTrackbar("LowV", "Control", &iLowV,255); //Control Value
	cvCreateTrackbar("HighV", "Control", &iHighV,255);

	while(true)
	{
		Mat imgOriginal; 
		bool bSuccess = cap.read(imgOriginal); 
		if(!bSuccess) 
		{
			cout << "Cannot read frame from Video Stream" << endl;; 
			break;
		}
		Mat imgHSV; 
		cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); 
		Mat imgThreshold; 
		inRange(imgHSV, Scalar(iLowH,iLowS,iLowV), Scalar(iHighH,iHighS, iHighV), imgThreshold); 
		 
		//Erode and Dilate functions removes objects in foreground and background that aren't the object we're focusing on
		erode(imgThreshold, imgThreshold, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
		dilate( imgThreshold, imgThreshold, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 

		dilate( imgThreshold, imgThreshold, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 
		erode(imgThreshold, imgThreshold, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

		imshow("Thresholded Image", imgThreshold); 
		imshow("Original",imgOriginal); 

		if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
        {
            cout << "esc key is pressed by user" << endl;
            break; 
         }
		
	}
	cvDestroyAllWindows();
	return 0; 

}
//Tracks Quadrilaterals, Triangles, and Hexagons using contours 
int tutorial10()
{
	 IplImage* img = cvLoadImage("FindingContours.png"); 

	 cvNamedWindow("RAW"); 
	 cvShowImage("RAW",img); 

	  IplImage* imgGrayScale = cvCreateImage(cvGetSize(img), 8,1); 
	  cvCvtColor(img,imgGrayScale,CV_BGR2GRAY); 

	  cvThreshold(imgGrayScale,imgGrayScale,128,255,CV_THRESH_BINARY);  

	  CvSeq* contours; //pointer to contour in memory block
	  CvSeq* result; //sequence of points of a contour 
	  CvMemStorage *storage = cvCreateMemStorage(0); //Storage Area for contours 

	  cvFindContours(imgGrayScale, storage, &contours, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));

	  while(contours)
	  {
		  result = cvApproxPoly(contours,sizeof(CvContour),storage, CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.02,0); 
		  //Find Triangles
		  if(result ->total == 3)
		  {
			  CvPoint *pt[3]; 
			  for(int i =0; i < 3; i++)
			  {
				  pt[i] = (CvPoint*)cvGetSeqElem(result,i); 
			  }

			  cvLine(img, *pt[0], *pt[1], cvScalar(255,0,0), 4);
			  cvLine(img, *pt[1], *pt[2], cvScalar(255,0,0), 4);
			  cvLine(img, *pt[2], *pt[0], cvScalar(255,0,0), 4);

		  }
		  //Finds Quadrilaterals 
		else if(result->total==4 )
		{
			 CvPoint *pt[4];
			 for(int i=0;i<4;i++)
			 {
             pt[i] = (CvPoint*)cvGetSeqElem(result, i);
			 }
   
			 cvLine(img, *pt[0], *pt[1], cvScalar(0,255,0),4);
			 cvLine(img, *pt[1], *pt[2], cvScalar(0,255,0),4);
			 cvLine(img, *pt[2], *pt[3], cvScalar(0,255,0),4);
			 cvLine(img, *pt[3], *pt[0], cvScalar(0,255,0),4);   
			}
			//Finds hexagons
			else if(result->total ==7  )
			{
			 CvPoint *pt[7];
			 for(int i=0;i<7;i++)
			 {
				 pt[i] = (CvPoint*)cvGetSeqElem(result, i);
			 }
   
			 cvLine(img, *pt[0], *pt[1], cvScalar(0,0,255),4);
			 cvLine(img, *pt[1], *pt[2], cvScalar(0,0,255),4);
			 cvLine(img, *pt[2], *pt[3], cvScalar(0,0,255),4);
			 cvLine(img, *pt[3], *pt[4], cvScalar(0,0,255),4);
			 cvLine(img, *pt[4], *pt[5], cvScalar(0,0,255),4);
			 cvLine(img, *pt[5], *pt[6], cvScalar(0,0,255),4);
			 cvLine(img, *pt[6], *pt[0], cvScalar(0,0,255),4);
		 }

		 contours = contours->h_next; 
		}
	  cvNamedWindow("Tracked"); 
	  cvShowImage("Tracked",img); 
	  cvWaitKey(0); 

	  cvDestroyAllWindows(); 
	  cvReleaseMemStorage(&storage);
	  cvReleaseImage(&img);
	  cvReleaseImage(&imgGrayScale);

	  return 0; 
}

//Calls the various tutorial functions and allows all of the tutorials to be in one file instead of 10
int main()
{
	int exit1 = tutorial1(); 
	int exit2 = tutorial2(); 
	int exit3 = tutorial3(); 
	int exit4 = tutorial4();
	int exit5 = tutorial5(); 
	int exit6 = tutorial6(); 
	int exit7 = tutorial7(); 
	int exit8 = tutorial8(); 
	int exit9 = tutorial9(); 
	int exit10 = tutorial10(); 

	return 0; 
}

