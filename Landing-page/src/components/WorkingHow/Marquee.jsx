import React from "react";
import Marquee from "react-fast-marquee";
import "./Marquee.css";

const techIcons = [
  {
    src: "https://upload.wikimedia.org/wikipedia/commons/a/a7/React-icon.svg",
    alt: "React",
  },
  {
    src: "https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg",
    alt: "Python",
  },
  {
    src: "https://upload.wikimedia.org/wikipedia/commons/d/d9/Node.js_logo.svg",
    alt: "Node.js",
  },
  {
    src: "https://upload.wikimedia.org/wikipedia/commons/6/62/CSS3_logo.svg",
    alt: "CSS",
  },
  {
    src: "https://upload.wikimedia.org/wikipedia/commons/6/61/HTML5_logo_and_wordmark.svg",
    alt: "HTML",
  },
  {
    src: "https://upload.wikimedia.org/wikipedia/commons/8/8a/Google_Gemini_logo.svg",
    alt: "Gemini API",
  },
  {
    src: "https://upload.wikimedia.org/wikipedia/commons/7/77/Streamlit-logo-primary-colormark-darktext.png",
    alt: "Streamlit",
  },
  {
    src: "https://upload.wikimedia.org/wikipedia/commons/2/2d/Tensorflow_logo.svg",
    alt: "TensorFlow",
  },
  {
    src: "https://pypi.org/static/images/logo-small.8998e9d1.svg",
    alt: "PyPI",
  },
];

const MarqueeComponent = () => {
  return (
    <div className="marquee-container">
      <Marquee speed={60} pauseOnHover={true} gradient={false}>
        {techIcons.map((tech, index) => (
          <div key={index} className="icon-container">
            <img src={tech.src} alt={tech.alt} className="icon" />
          </div>
        ))}
      </Marquee>
    </div>
  );
};

export default MarqueeComponent;
