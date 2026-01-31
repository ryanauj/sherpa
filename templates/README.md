# Templates

This directory contains templates for creating paired tutorials and guides. The dual structure ensures content works effectively in all three learning modes.

## Available Templates

### tutorial-template.md (Human-Focused)
Use this template for creating content that humans read independently. Tutorials should:
- Explain concepts clearly in readable prose
- Include detailed examples with explanations
- Provide exercises with expandable hints and solutions
- Be self-contained and understandable without external help
- Guide learners progressively through the topic

**When to use:** Creating the human-readable learning material

### guide-template.md (AI-Focused)
Use this template for creating teaching scripts for AI assistants. Guides should:
- Provide structured teaching flow with timing
- Include specific questions to ask learners
- List common misconceptions and how to address them
- Offer adaptive strategies for different skill levels
- Give guidance on how to present exercises and help with them

**When to use:** Creating the AI teaching script

## The Pairing Concept

Each topic should have BOTH a tutorial and a guide:
- Same topic, same filename
- Different directories (`tutorials/` vs `guides/`)
- Aligned content but optimized for different users

**Example:**
```
tutorials/tools/git-basics.md    ← Humans read this
guides/tools/git-basics.md       ← AI assistants use this
```

## How to Create Paired Content

### Step 1: Plan Your Topic
Decide what you want to teach:
- What are the learning objectives?
- What concepts need to be covered?
- What exercises will reinforce learning?

### Step 2: Create the Tutorial First
```bash
cp templates/tutorial-template.md tutorials/[category]/[topic-name].md
```

Fill in:
- Clear explanations a human can read
- Detailed code examples with walkthroughs
- Exercises with hints and solutions
- Practice project

### Step 3: Create the Paired Guide
```bash
cp templates/guide-template.md guides/[category]/[topic-name].md
```

Fill in:
- How to teach each concept
- Questions to ask to verify understanding
- Common misconceptions to watch for
- How to guide through the exercises
- Adaptive strategies

### Step 4: Cross-Reference
In both files, update the YAML front matter:
- Tutorial: `paired_guide: /guides/[category]/[topic-name].md`
- Guide: `paired_tutorial: /tutorials/[category]/[topic-name].md`

### Step 5: Align Content
Make sure:
- Both cover the same concepts in the same order
- Exercises in the guide match exercises in the tutorial
- Section names are similar (doesn't have to be exact)
- Learning objectives are identical

## Tips for Creating Great Content

### For Tutorials
- Write as if teaching a friend
- Explain "why" not just "how"
- Use analogies and real-world examples
- Make examples runnable and practical
- Include self-check points

### For Guides
- Think like a teacher planning a lesson
- Include timing estimates for each section
- Anticipate where learners will struggle
- Provide multiple ways to explain concepts
- Include questions that check understanding

### For Both
- Start simple, build complexity gradually
- Use consistent terminology
- Test your content with real learners
- Iterate based on feedback

## Quality Checklist

Before submitting paired content, verify:

**Tutorial:**
- [ ] Clear learning objectives
- [ ] Readable explanations without jargon (or jargon explained)
- [ ] Working code examples
- [ ] Exercises with hints and solutions
- [ ] Self-check points throughout
- [ ] Practice project with requirements
- [ ] References paired guide in front matter

**Guide:**
- [ ] Teaching flow with time estimates
- [ ] Verification questions at each stage
- [ ] Common misconceptions addressed
- [ ] Adaptive strategies for different learners
- [ ] Exercise guidance (not just answers)
- [ ] References paired tutorial in front matter

**Alignment:**
- [ ] Same learning objectives
- [ ] Same concepts covered in same order
- [ ] Exercises match between tutorial and guide
- [ ] Same examples or complementary ones

## Examples

See these examples of paired content:
- `tutorials/tools/git-basics.md` and `guides/tools/git-basics.md`

## Questions?

If you're unsure about:
- **What to put in tutorial vs guide**: Tutorial = content to read, Guide = how to teach
- **How detailed to be**: Tutorial = very detailed, Guide = teaching strategy + key points
- **Whether content aligns**: Have someone review using one while you use the other

See `/CONTRIBUTING.md` for more detailed guidelines, or open an issue to ask!
