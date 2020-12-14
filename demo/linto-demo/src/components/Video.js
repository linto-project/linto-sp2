import { useEffect, useRef } from "react";
import "./Video.css";

const Video = ({ url, isPlaying, durationSec, setVideoLoaded }) => {
  const video = useRef(null);

  useEffect(() => {
    video.current.currentTime = durationSec;
  }, [durationSec]);

  useEffect(() => {
    isPlaying ? video.current.play() : video.current.pause();
  }, [isPlaying]);

  const videoReady = () => {
    setVideoLoaded(true);
  };

  return (
    <div className="video">
      <video
        preload
        //   ref={ref}
        ref={video}
        src={url}
        width="640"
        height="360"
        onCanPlayThrough={videoReady}
        //   onTimeUpda te={onSetVideoTimestamp}
      />
    </div>
  );
};

export default Video;
