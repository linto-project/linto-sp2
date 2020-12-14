import { Button as MuiButton } from "@material-ui/core";
import "./Button.css";
import { makeStyles, createStyles } from "@material-ui/core/styles";
import clsx from "clsx";

const useStyles = makeStyles(() =>
  createStyles({
    root: {
      width: "300px",
      border: "2px solid black",
    },

    selected: {
      border: "2px solid white",
    },
  })
);

const Button = ({ description, selected, children, ...rest }) => {
  const classes = useStyles();
  return (
    <div>
      <MuiButton
        variant="contained"
        className={
          selected ? clsx(classes.root, classes.selected) : classes.root
        }
        {...rest}
      >
        {children}
      </MuiButton>
    </div>
  );
};

export default Button;
