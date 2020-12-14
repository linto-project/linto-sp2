import "./App.css";

// import Demo from "./pages/Demo";
// import DemoVideo from "./pages/DemoVideo";
// import DemoAudio from "./pages/DemoAudio";
import DemoVideoAudio from "./pages/DemoVideoAudio";

// import { BrowserRouter as Router, Switch, Route, Link } from "react-router-dom";
import { BrowserRouter as Router } from "react-router-dom";

const App = () => {
  return (
    <Router>
      <div>
        <DemoVideoAudio />
      </div>
    </Router>
  );
};

export default App;
