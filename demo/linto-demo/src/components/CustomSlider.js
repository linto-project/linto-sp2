import Slider from "@material-ui/core/Slider";
import Grid from "@material-ui/core/Grid";
import Box from "@material-ui/core/Box";

import Typography from "@material-ui/core/Typography";

import "./CustomSlider.css";

const CustomeSlider = ({
  id,
  name,
  value,
  onChange,
  min,
  max,
  step,
  children,
  icon,
  disabled,
}) => {
  return (
    <Box width="100%" display="inline-flex" flexDirection="column">
      <Typography id="input-slider" gutterBottom>
        {children}
      </Typography>
      <Grid container spacing={2}>
        <Grid item>{icon}</Grid>
        <Grid xs item>
          <Slider
            disabled={disabled}
            id={id}
            name={name}
            value={value}
            onChange={onChange}
            aria-labelledby="continuous-slider"
            min={min}
            max={max}
            step={step}
            valueLabelDisplay="auto"
          />
        </Grid>
      </Grid>
    </Box>
  );
};
export default CustomeSlider;
