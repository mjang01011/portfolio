import "./About.css";
import brush_skyblue from "../../assets/brush_skyblue.png";

const About = () => {
  return (
    <div id="about" className="about">
      <div className="about-title">
        <h1>About Me</h1>
        <img src={brush_skyblue} alt="" />
      </div>
      <div className="about-sections">
        {/* <div className="about-left"> */}
        {/* <img src={profile_img} alt="" /> */}
        {/* </div> */}
        <div className="about-right">
          <div className="about-para">
            <p>
              I am a passionate technologist with a keen interest in the wonders
              of both the digital and physical worlds. My journey spans the
              realms of machine learning, software development,
              astrophotography, and even the art of magic. Prior to my
              engineering studies, I was involved in astrophysics research experiences.
            </p>
            <p>
              I thrive in environments where collaboration, continuous learning,
              and development are encouraged, and I am always seeking
              opportunities to expand my skillset by exploring and integrating
              new tools and methodologies into my work.
            </p>
            <p>
              Apart from studies, you will often find me gazing at the night sky
              with my telescope. Astrophotography allows me to combine my love
              for technology and the cosmos, capturing the beauty of distant
              deep sky objects. It’s a humbling experience that constantly
              reminds me of the vastness of the universe and our small, yet
              significant place within it. Magic has also been a source of
              wonder and joy in my life. Performing magic tricks is more than
              just a hobby; it’s a way to connect with people and bring a sense
              of awe and excitement into their lives.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default About;
