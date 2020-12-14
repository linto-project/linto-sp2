import "./Rectangle.css";
import { makeStyles } from "@material-ui/core/styles";

const useStyles = makeStyles({
  // style rule
  root: (props) => ({
    top: props.top + "px",
    left: props.left + "px",
    width: props.width + "px",
    height: props.height + "px",
  }),
});

const Rectangle = (props) => {
  const classes = useStyles(props);
  console.log("props : ");
  console.log(props);
  return <div className={`rectangle ${classes.root}`} />;
};

Rectangle.defaultProps = {
  top: 50,
  left: 200,

  width: 200,
  height: 200,
};

export default Rectangle;
