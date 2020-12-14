import "./VideoAudio.css";
import { useState, useEffect } from "react";
import Video from "./Video";
import WaveSurfer from "./Waveform";

import Grid from "@material-ui/core/Grid";
import CustomeSlider from "./CustomSlider";
import Button from "./Button";
import CircularProgress from "@material-ui/core/CircularProgress";
import Backdrop from "@material-ui/core/Backdrop";

import VolumeUp from "@material-ui/icons/VolumeUp";
import ZoomOut from "@material-ui/icons/ZoomOut";
import PlayArrowSharpIcon from "@material-ui/icons/PlayArrowSharp";
import StopSharpIcon from "@material-ui/icons/StopSharp";

const VideoAudio = ({ framerate }) => {
  console.log("framerate  : " + framerate);

  const [durationSec, setDurationSec] = useState([]);
  const [isPlaying, setIsPlaying] = useState([]);
  const [frame, setFrame] = useState([]);
  const [play, setPlay] = useState(false);
  const [synch, setSynch] = useState(false);
  const [volume, setVolume] = useState(0.5);
  const [zoom, setZoom] = useState(5);

  const [audioLoaded, setAudioLoaded] = useState(false);
  const [videoLoaded, setVideoLoaded] = useState(false);
  const [playDisabled, setPlayDisabled] = useState(true);

  const [open, setOpen] = useState(false);

  const handleClose = () => {
    setOpen(false);
  };

  useEffect(() => {
    console.log("current State : " + isPlaying);
  }, [isPlaying]);

  useEffect(() => {
    console.log("current timestamp : " + durationSec);
    setFrame(Math.round(durationSec * 25));
  }, [durationSec]);

  useEffect(() => {
    audioLoaded && videoLoaded ? setPlayDisabled(false) : setPlayDisabled(true);
    audioLoaded && videoLoaded ? setOpen(false) : setOpen(true);
  }, [audioLoaded, videoLoaded]);

  const handlePlayPause = () => {
    if (!playDisabled) {
      setPlay(!play);
    }
  };

  const onVolumeChange = (e, newValue) => {
    if (newValue) {
      setVolume(newValue);
    }
  };

  const onZoomChange = (e, newValue) => {
    if (newValue) {
      setZoom(newValue);
    }
  };

  return (
    <div className="demo">
      {/* @to do:
        Modifiy backdrop for skeleton (material)
      */}
      <Backdrop className="backdrop" open={open} onClick={handleClose}>
        <CircularProgress color="inherit" />
        <p> Video & Audio loading, please wait </p>
      </Backdrop>
      <Grid container direction="column" spacing={2}>
        <Grid item justify="center">
          <Video
            url="/video/IS1002b.Closeup1-1-5min.webm"
            durationSec={durationSec}
            isPlaying={isPlaying}
            setVideoLoaded={setVideoLoaded}
          />
        </Grid>
        <Grid item>
          <div id="waveform">
            <WaveSurfer
              url="/audio/IS1002b.Array1-01-5min.wav"
              zoom={zoom}
              synch={synch}
              setSynch={setSynch}
              play={play}
              setPlay={setPlay}
              volume={volume}
              audioLoaded={audioLoaded}
              setAudioLoaded={setAudioLoaded}
              playDisabled={playDisabled}
              setPlayDisabled={setPlayDisabled}
              setIsPlaying={setIsPlaying}
              setDurationSec={setDurationSec}
              setFrame={setFrame}
              framerate={framerate}
            />
          </div>
        </Grid>
        <Grid item>
          <div>
            <Grid container spacing={2} alignItems="center">
              <Grid item align-items="baseline">
                <div className="control">
                  <Button
                    style={{ maxWidth: "100px", minWidth: "100px" }}
                    variant="contained"
                    color="primary"
                    onClick={handlePlayPause}
                    fullWidth="false"
                    disabled={playDisabled}
                    startIcon={
                      !play ? <PlayArrowSharpIcon /> : <StopSharpIcon />
                    }
                  >
                    {!play ? "Play" : "Pause"}
                  </Button>
                </div>
              </Grid>
              <Grid xs item>
                <CustomeSlider
                  id="volume"
                  name="volume"
                  value={volume}
                  onChange={onVolumeChange}
                  aria-labelledby="continuous-slider"
                  min={0}
                  max={1}
                  step={0.01}
                  icon={<VolumeUp />}
                  valueLabelDisplay="auto"
                >
                  Volume
                </CustomeSlider>
              </Grid>
              <Grid xs item>
                <CustomeSlider
                  id="zoom"
                  name="zoom"
                  value={zoom}
                  onChange={onZoomChange}
                  aria-labelledby="continuous-slider"
                  min={1}
                  max={10}
                  step={0.1}
                  icon={<ZoomOut />}
                  valueLabelDisplay="auto"
                >
                  Zoom
                </CustomeSlider>
              </Grid>
            </Grid>
          </div>
        </Grid>
        <Grid item>
          <div>
            <p style={{ fontSize: "40px" }}>Current frame: {frame}</p>
          </div>
        </Grid>
      </Grid>
    </div>
  );
};

export default VideoAudio;
