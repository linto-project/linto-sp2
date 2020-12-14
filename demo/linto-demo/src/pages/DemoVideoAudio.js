import "./Demo.css";

import VideoAudio from "../components/VideoAudio";
import IHM from "../components/IHM";
import { Grid } from "@material-ui/core";

const DemoVideoAudio = () => {
  return (
    <div className="demo">
      <Grid container>
        <Grid item xs={2}>
          <IHM />
        </Grid>
        <Grid item xs={10}>
          <VideoAudio framerate={25} />
        </Grid>
      </Grid>
    </div>
  );
};

export default DemoVideoAudio;
