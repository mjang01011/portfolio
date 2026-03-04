import "./About.css";
import { useState } from "react";
import sail_logo from "../../assets/sail_logo.jpg";
import duke_logo from "../../assets/duke_logo.png";
import nvidia_logo from "../../assets/nvidia_logo.png";
import google_logo from "../../assets/google_logo.png";
import amd_logo from "../../assets/amd_logo.png";

const orgLogos = {
  "Stanford Vision and Learning Lab (SVL)": sail_logo,
  "Advised by Dr. Lawrence Carin, Duke University": duke_logo,
  "aiMIND Study Group, Duke University": duke_logo,
  NVIDIA: nvidia_logo,
  Google: google_logo,
  "AMD (Advanced Micro Devices, Inc.)": amd_logo,
  "Duke University": duke_logo,
};

const experiences = [
  {
    title: "Machine Learning Researcher",
    location: "Stanford, CA, USA",
    org: "Stanford Vision and Learning Lab (SVL)",
    dates: "Jan 2026 – Current",
    incoming: false,
    details: [
      "Developing new visual generation dataset and benchmarking state-of-the-art image generation models.",
    ],
  },
  {
    title: "Machine Learning Researcher",
    location: "Durham, NC, USA",
    org: "Advised by Dr. Lawrence Carin, Duke University",
    dates: "Jan 2025 – May 2025",
    incoming: false,
    details: [
      "Developed a scalable Bag-of-Words topic model that learns interpretable low-dimensional document embeddings.",
      "Extended the model to integrate document-level question-answer data, designing a double-softmax latent factor model to identify narratives that explain structured QA responses across documents.",
    ],
  },
  {
    title: "Machine Learning Researcher",
    location: "Durham, NC, USA",
    org: "aiMIND Study Group, Duke University",
    dates: "Mar 2024 – May 2025",
    incoming: false,
    details: [
      "Led the development of a multimodal deep learning model in PyTorch to predict neurodegenerative disease progression, and built scalable pipelines to process 6,000+ retinal images for end-to-end training.",
      "Achieved AUC of 0.985 on Mild Cognitive Impairment prediction with a fused multimodal EfficientNet-B6.",
    ],
  },
  {
    title: "Incoming Software Engineering Intern",
    location: "",
    org: "NVIDIA",
    dates: "Incoming Fall 2026",
    incoming: true,
    details: ["GPU Compute Software QA"],
  },
  {
    title: "Incoming Software Engineering Intern",
    location: "",
    org: "Google",
    dates: "Incoming Summer 2026",
    incoming: true,
    details: ["GCP Networking, LLMs"],
  },
  {
    title: "Software Engineering Intern",
    location: "Austin, TX, USA",
    org: "AMD (Advanced Micro Devices, Inc.)",
    dates: "June 2025 – September 2025",
    incoming: false,
    details: [
      "Designed an agentic system using FastMCP and RAG to interface a production-ready LLM-powered assistant with internal triage tools, accelerating silicon log analysis and automated failure triage for next-generation server CPUs.",
      "Spearheaded the development of a LLM-powered debugging assistant that translates engineers' natural language requests into optimized MongoDB queries for efficient data retrieval and diagnostics.",
      "Established a self-hosted MongoDB sync pipeline from Cosmos DB using Python and GitHub Actions, achieving faster access latency, reduced API costs, and elimination of rate limiter failures through read/write optimization.",
    ],
  },
  {
    title: "Teaching Assistant",
    location: "Durham, NC, USA",
    org: "Duke University",
    dates: "2024 Fall – 2025 Spring",
    incoming: false,
    details: [
      "2025 Spring: ECE580 Introduction to Machine Learning (Graduate). Key concepts include discriminative/generative classifiers (SVM, RVM, logistic regression, Bayes), dimensionality reduction, feature selection, and performance evaluation.",
      "2024 Fall: ECE480 Applied Probability for Statistical Learning. Key concepts include Bayesian inference, probabilistic reasoning, mixture models, and model selection.",
      "2024 Fall: CS371 Elements of Machine Learning. Key concepts include linear/nonlinear SVM, kernels, decision trees, CNNs, and transformers.",
    ],
  },
];

const ExperienceCard = ({ experience, isExpanded, onToggle }) => {
  return (
    <div
      className={`timeline-card ${isExpanded ? "expanded" : ""} ${experience.incoming ? "incoming" : ""}`}
      onClick={onToggle}
    >
      <div className="timeline-card-header">
        <div className="timeline-card-header-text">
          <h4 className="timeline-title">{experience.title}</h4>
          <p className="timeline-org">{experience.org}</p>
          <p className="timeline-dates">{experience.dates}</p>
        </div>
        <div className="timeline-header-right">
          {orgLogos[experience.org] && (
            <img src={orgLogos[experience.org]} alt={experience.org} className="timeline-org-logo" />
          )}
          <span className="timeline-expand-icon">{isExpanded ? "−" : "+"}</span>
        </div>
      </div>
      <div className={`timeline-details-wrapper ${isExpanded ? "expanded" : ""}`}>
        <ul className="timeline-details">
          {experience.details.map((detail, i) => (
            <li key={i}>{detail}</li>
          ))}
        </ul>
      </div>
    </div>
  );
};

const About = () => {
  const [expandedId, setExpandedId] = useState(null);

  const researchExperiences = experiences.slice(0, 3);
  const experiencesList = experiences.slice(3, 7);

  return (
    <section id="about" className="about">
      <div className="section-inner">
        <h2 className="section-title">About Me</h2>
        <div className="about-intro">
          <p>
            I am passionate about <strong>multimodal large language models</strong>: models that understand and generate across text, images, and other modalities.
          </p>
          <p>
            I thrive in collaborative research environments and enjoy integrating new tools and methods into my work. Outside of research, I love travel photography and astrophotography.
          </p>
        </div>

        <div className="timeline-wrapper">
          <div className="timeline-section">
            <h3 className="timeline-section-title">Research</h3>
            <div className="timeline-scroll">
              {researchExperiences.map((exp, i) => (
                <ExperienceCard
                  key={i}
                  experience={exp}
                  isExpanded={expandedId === i}
                  onToggle={(e) => {
                    e.stopPropagation();
                    setExpandedId(expandedId === i ? null : i);
                  }}
                />
              ))}
            </div>
          </div>
          <div className="timeline-section">
            <h3 className="timeline-section-title">Experiences</h3>
            <div className="timeline-scroll">
              {experiencesList.map((exp, i) => (
                <ExperienceCard
                  key={3 + i}
                  experience={exp}
                  isExpanded={expandedId === 3 + i}
                  onToggle={(e) => {
                    e.stopPropagation();
                    setExpandedId(expandedId === 3 + i ? null : 3 + i);
                  }}
                />
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default About;
