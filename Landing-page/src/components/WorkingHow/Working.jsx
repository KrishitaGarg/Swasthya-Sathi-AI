import { FaFileUpload, FaBrain, FaFileDownload, FaRobot } from "react-icons/fa";
import "./Working.css";
import Marquee from "./Marquee";

const steps = [
  {
    icon: <FaFileUpload />,
    title: "Upload Medical Documents",
    description: "Securely upload your medical files for AI analysis.",
  },
  {
    icon: <FaBrain />,
    title: "AI Analyzes Your Files",
    description: "Our AI scans your documents for insights and diagnoses.",
  },
  {
    icon: <FaFileDownload />,
    title: "Download Detailed Report",
    description: "Get an easy-to-read report with AI-driven recommendations.",
  },
  {
    icon: <FaRobot />,
    title: "Chat with AI HelpBot",
    description: "Ask questions and get AI-powered medical insights instantly.",
  },
];

export default function HowItWorks() {
  return (
    <section id="working" className="how-it-works">
      <h2>How It Works</h2>
      <p>Simple and AI-powered healthcare assistance in four easy steps.</p>
      <div className="steps-grid">
        {steps.map((step, index) => (
          <div className="step-card" key={index}>
            <div className="step-icon">{step.icon}</div>
            <h3 className="step-title">{step.title}</h3>
            <p className="step-description">{step.description}</p>
          </div>
        ))}
      </div>
      <Marquee />
    </section>
  );
}
