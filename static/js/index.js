//Function for selecting different video sources
function select_videos(option)
{
  //var option = document.getElementById("videos").value;

  if(option === 'local_inference')
  {
    document.getElementById("image").src = "/local_video_inference";

    document.getElementById("image1").src = "/local_video_inference_trk";
  }
  else if(option === 'local_video')
  {
    document.getElementById("image").src = "/local_video_live";

    document.getElementById("image1").src = "/local_video_live_trk";
  }
  else if(option === 'webcam')
  {
    document.getElementById("image").src = "/video_webcam";

    document.getElementById("image1").src = "/video_webcam_trk";
  }
  else if(option === 'video_live_golf')
  {
    document.getElementById("image").src = "/video_live_golf";

    document.getElementById("image1").src = "/video_live_golf_trk";
  }

  /*
  if(option === 'live_webcam')
  {
    document.getElementById("testing").innerHTML = "src changes to live webcam";
  }
  else
  {
    document.getElementById("testing").innerHTML = "src changes to local video";
  }
  */
  isCalled = false; //global variable
  removeImg();
}

//function for removing the multipe videos
function removeImg()
{
  var elements = document.getElementsByClassName("detect_img");
  for (var i = elements.length - 1; i > 0; i--)
  {
    elements[i].parentNode.removeChild(elements[i]);
  }

  for (var i = 0; i < elements.length; i++) {
    elements[i].style.cssFloat = "";
    elements[i].style.width = "";
  }

  var elements2 = document.getElementsByClassName("track_img");
  for (var i = elements2.length - 1; i > 0; i--)
  {
    elements2[i].parentNode.removeChild(elements2[i]);
  }

  for (var i = 0; i < elements2.length; i++) {
    elements2[i].style.cssFloat = "";
    elements2[i].style.width = "";
  }
}

//global variable for enabling to call the multiple video once only
var isCalled = false;

//Function for selecting multipe Videos
function multiple_videos()
{
    if(isCalled == false)
    {
      //for detect components
      var img0 = document.getElementById("image");
      img0.src = "/video_terrace_c0";
      img0.className = "detect_img";

      var img1 = document.createElement("img");
      img1.src = "/video_terrace_c1";
      img1.className = "detect_img";

      var img2 = document.createElement("img");
      img2.src = "/video_terrace_c2";
      img2.className = "detect_img";

      var img3 = document.createElement("img");
      img3.src = "/video_terrace_c3";
      img3.className = "detect_img";

      document.getElementById("detect_block").appendChild(img1);
      document.getElementById("detect_block").appendChild(img2);
      document.getElementById("detect_block").appendChild(img3);

      var selector = document.getElementsByClassName("detect_img");
      for (var i = 0; i < selector.length; i++) {
        selector[i].style.cssFloat = "left";
        selector[i].style.width = "50%";
      }

      //for track components
      var img03 = document.getElementById("image1");
      img03.src = "/video_terrace_c0_trk";
      img03.className = "track_img";

      var img4 = document.createElement("img");
      img4.src = "/video_terrace_c1_trk";
      img4.className = "track_img";

      var img5 = document.createElement("img");
      img5.src = "/video_terrace_c2_trk";
      img5.className = "track_img";

      var img6 = document.createElement("img");
      img6.src = "/video_terrace_c3_trk";
      img6.className = "track_img";

      document.getElementById("track_block").appendChild(img4);
      document.getElementById("track_block").appendChild(img5);
      document.getElementById("track_block").appendChild(img6);

      var selector2 = document.getElementsByClassName("track_img");
      for (var i = 0; i < selector2.length; i++) {
        selector2[i].style.cssFloat = "left";
        selector2[i].style.width = "50%";
      }

      isCalled = true;
    }


}
